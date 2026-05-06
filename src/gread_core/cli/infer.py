"""Inference CLI: run LLM-free predictions with a trained GReaD-Core model.

Usage:
    python -m gread_core.cli.infer --config configs/default.yaml \
        --dataset tiny --detector gcn \
        --detector-checkpoint artifacts/detector/ \
        --reasoner-checkpoint artifacts/reasoner/ \
        --output-dir artifacts/predictions/
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from gread_core.detectors.factory import (
    ALL_DETECTOR_TYPES,
    create_detector,
    get_detector_embedding_dim,
)
from gread_core.experiment.seed import set_seed
from gread_core.inference.predictor import GReaDInferencePipeline
from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.reasoner import GReaDReasoner


def _load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _load_detector(
    config: dict[str, Any],
    detector_name: str,
    checkpoint_path: Path,
    device: torch.device,
    in_channels: int,
) -> torch.nn.Module:
    """Instantiate and load a base detector from checkpoint."""
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)
    detector = create_detector(
        detector_name,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        config=config,
    )

    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        detector.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )

    detector.to(device)
    detector.eval()
    return detector


def _load_reasoner(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> GReaDReasoner:
    """Instantiate and load a GReaDReasoner from checkpoint."""
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)
    num_evidence_slots = config.get("evidence", {}).get("num_slots", 16)
    rho = config.get("method", {}).get("residual_rho", 0.1)
    signed_evidence_masks = config.get("method", {}).get("signed_evidence_masks", True)

    evidence_encoder = EvidenceEncoder(
        vocab_size=num_evidence_slots + 10,
        embed_dim=32,
        num_slots=num_evidence_slots,
        output_dim=128,
    )

    reasoner = GReaDReasoner(
        hidden_dim=get_detector_embedding_dim(
            str(config.get("detector", {}).get("type", "gcn")),
            hidden_channels=hidden_channels,
            config=config,
        ),
        evidence_encoder=evidence_encoder,
        num_risk_types=6,
        num_evidence_slots=num_evidence_slots,
        rho=rho,
        signed_evidence_masks=signed_evidence_masks,
    )

    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        reasoner.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=device)
        )

    reasoner.to(device)
    reasoner.eval()
    return reasoner


def main(argv: list[str] | None = None) -> None:
    """Inference CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GReaD-Core LLM-Free Inference"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (e.g. yelpchi, amazon, tiny)",
    )
    parser.add_argument(
        "--detector", type=str, default="gcn",
        choices=ALL_DETECTOR_TYPES,
        help="Detector architecture name",
    )
    parser.add_argument(
        "--detector-checkpoint", type=str, required=True,
        help="Path to detector checkpoint directory",
    )
    parser.add_argument(
        "--reasoner-checkpoint", type=str, required=True,
        help="Path to reasoner checkpoint directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts/predictions",
        help="Directory for output files",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on (cpu, cuda, cuda:0, ...)",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Real dataset root; overrides config data.root and GREAD_DATA_ROOT",
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--experiment-id", type=str, default="infer",
        help="Experiment identifier",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    set_seed(args.seed)

    config = _load_config(args.config)
    config.setdefault("detector", {})["type"] = args.detector
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset.
    from gread_core.data.loaders import load_graph_dataset
    data = load_graph_dataset(
        args.dataset, root=args.data_root, seed=args.seed, config=config
    )
    data = data.to(device)

    # Build detector.
    logger.info("Loading detector '%s' from %s", args.detector, args.detector_checkpoint)
    detector = _load_detector(
        config=config,
        detector_name=args.detector,
        checkpoint_path=Path(args.detector_checkpoint),
        device=device,
        in_channels=data.x.shape[1],
    )

    # Build reasoner.
    logger.info("Loading reasoner from %s", args.reasoner_checkpoint)
    reasoner = _load_reasoner(
        config=config,
        checkpoint_path=Path(args.reasoner_checkpoint),
        device=device,
    )

    # Build adapter using detector forward pass on full graph (no masks).
    from gread_core.adapters.factory import create_evidence_adapter

    inference_graph = copy.copy(data)
    for attr in ("train_mask", "val_mask", "test_mask"):
        if hasattr(inference_graph, attr):
            setattr(inference_graph, attr, None)

    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(inference_graph)  # type: ignore[operator]
    adapter_thresholds = config.get("adapter", {}).get("thresholds", {})
    adapter = create_evidence_adapter(
        detector_type=args.detector,
        detector=detector,
        graph=inference_graph,
        logits=logits,
        embeddings=embeddings,
        thresholds=adapter_thresholds,
        strict_detector_signal=args.dataset != "tiny",
    )

    # Build pipeline.
    pipeline = GReaDInferencePipeline(
        detector=detector,
        reasoner=reasoner,
        adapter=adapter,
        config=config,
    )

    # Determine target nodes (test nodes if available, else all).
    if hasattr(data, "test_mask") and data.test_mask is not None and data.test_mask.any():
        node_ids = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    else:
        node_ids = list(range(data.x.shape[0]))

    logger.info("Running inference on %d nodes", len(node_ids))

    # Run predictions.
    predictions = pipeline.predict(data, node_ids)
    logger.info("Generated %d predictions", len(predictions))

    # Save results.
    results = [asdict(p) for p in predictions]
    output_file = output_dir / f"{args.experiment_id}_predictions.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "experiment_id": args.experiment_id,
                "dataset": args.dataset,
                "detector": args.detector,
                "device": str(device),
                "seed": args.seed,
                "num_predictions": len(results),
                "predictions": results,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Predictions saved to %s", output_file)


if __name__ == "__main__":
    main()

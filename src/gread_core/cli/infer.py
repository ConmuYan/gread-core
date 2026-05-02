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
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

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
) -> torch.nn.Module:
    """Instantiate and load a base detector from checkpoint."""
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)
    in_channels = config.get("detector", {}).get("in_channels", 64)

    detector: torch.nn.Module
    if detector_name == "gcn":
        from gread_core.detectors.pyg_gnn import GCNDetector
        detector = GCNDetector(in_channels=in_channels, hidden_channels=hidden_channels)
    elif detector_name == "gat":
        from gread_core.detectors.pyg_gnn import GATDetector
        detector = GATDetector(in_channels=in_channels, hidden_channels=hidden_channels)
    elif detector_name == "bwgnn":
        from gread_core.detectors.bwgnn import BWGNNDetector
        detector = BWGNNDetector(in_channels=in_channels, hidden_channels=hidden_channels)
    elif detector_name == "caregnn":
        from gread_core.detectors.caregnn import CAREGNNDetector
        detector = CAREGNNDetector(in_channels=in_channels, hidden_channels=hidden_channels)
    elif detector_name == "tree_neighbor":
        from gread_core.detectors.tree_neighbor import TreeNeighborDetector
        detector = TreeNeighborDetector(in_channels=in_channels, hidden_channels=hidden_channels)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

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

    evidence_encoder = EvidenceEncoder(
        vocab_size=num_evidence_slots + 10,
        embed_dim=32,
        num_slots=num_evidence_slots,
        output_dim=128,
    )

    reasoner = GReaDReasoner(
        hidden_dim=hidden_channels,
        evidence_encoder=evidence_encoder,
        num_risk_types=6,
        num_evidence_slots=num_evidence_slots,
        rho=rho,
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
        choices=["gcn", "gat", "bwgnn", "caregnn", "tree_neighbor"],
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
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset.
    from gread_core.data.loaders import load_graph_dataset
    data = load_graph_dataset(args.dataset, seed=args.seed)
    data = data.to(device)

    # Build detector.
    logger.info("Loading detector '%s' from %s", args.detector, args.detector_checkpoint)
    detector = _load_detector(
        config=config,
        detector_name=args.detector,
        checkpoint_path=Path(args.detector_checkpoint),
        device=device,
    )

    # Build reasoner.
    logger.info("Loading reasoner from %s", args.reasoner_checkpoint)
    reasoner = _load_reasoner(
        config=config,
        checkpoint_path=Path(args.reasoner_checkpoint),
        device=device,
    )

    # Build adapter using detector forward pass on full graph.
    from gread_core.adapters.pyg_gnn_adapter import PyGGNNAdapter

    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(data)  # type: ignore[operator]
    adapter = PyGGNNAdapter(detector, data, logits, embeddings)

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

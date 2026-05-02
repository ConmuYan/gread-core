"""Stage 2 CLI: Generate ERRs via LLM.

CRITICAL: This is the ONLY stage that calls LLM.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from gread_core.experiment.seed import set_seed


def main(argv: list[str] | None = None) -> None:
    """Stage 2: Generate ERRs CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GReaD-Core Stage 2: Generate ERRs"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config path"
    )
    parser.add_argument(
        "--dataset", type=str, default="tiny", help="Dataset name"
    )
    parser.add_argument(
        "--detector", type=str, default="gcn",
        choices=["gcn", "gat", "bwgnn", "caregnn", "tree_neighbor"],
        help="Detector type",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Stage 1 checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts", help="Output dir"
    )
    parser.add_argument(
        "--experiment-id", type=str, default="stage2", help="Experiment ID"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--llm-backend", type=str, default="replay",
        choices=["replay", "openai", "stub"], help="LLM backend",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=".cache/llm", help="Cache dir"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    # Create experiment registry
    from gread_core.experiment.registry import ExperimentRegistry
    registry = ExperimentRegistry(
        experiment_id=args.experiment_id,
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Load dataset
    from gread_core.data.loaders import load_graph_dataset
    data = load_graph_dataset(args.dataset, seed=args.seed)

    # Create and load detector
    in_channels = data.x.shape[1]
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)

    detector: torch.nn.Module
    if args.detector == "gcn":
        from gread_core.detectors.pyg_gnn import GCNDetector
        detector = GCNDetector(
            in_channels=in_channels, hidden_channels=hidden_channels
        )
    elif args.detector == "gat":
        from gread_core.detectors.pyg_gnn import GATDetector
        detector = GATDetector(
            in_channels=in_channels, hidden_channels=hidden_channels
        )
    elif args.detector == "bwgnn":
        from gread_core.detectors.bwgnn import BWGNNDetector
        detector = BWGNNDetector(
            in_channels=in_channels, hidden_channels=hidden_channels
        )
    elif args.detector == "caregnn":
        from gread_core.detectors.caregnn import CAREGNNDetector
        detector = CAREGNNDetector(
            in_channels=in_channels, hidden_channels=hidden_channels
        )
    else:
        from gread_core.detectors.tree_neighbor import TreeNeighborDetector
        detector = TreeNeighborDetector(
            in_channels=in_channels, hidden_channels=hidden_channels
        )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        detector.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info("Loaded detector checkpoint from %s", model_path)

    # Create adapter (need logits + embeddings for ALL nodes)
    from gread_core.adapters.pyg_gnn_adapter import PyGGNNAdapter
    from gread_core.training.stage2_generate_err import _strip_masks
    no_mask_data = _strip_masks(data)
    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(no_mask_data)  # type: ignore[operator]
    adapter = PyGGNNAdapter(detector, data, logits, embeddings)

    # Create LLM client
    from gread_core.llm.clients import LLMClient
    if args.llm_backend == "stub":
        from gread_core.llm.clients import StubClient
        client: LLMClient = StubClient()
    elif args.llm_backend == "replay":
        from gread_core.llm.clients import ReplayClient
        client = ReplayClient(args.cache_dir)
    else:
        from gread_core.llm.clients import OpenAIClient
        client = OpenAIClient()

    # Create teacher and verifier
    from gread_core.llm.teacher import LLMTeacher
    from gread_core.verification.verifier import EvidenceContractVerifier

    contract_config = config.get("verifier", {})
    verifier = EvidenceContractVerifier(contract_config)
    teacher = LLMTeacher(client, verifier, args.cache_dir)

    # Generate ERRs
    from gread_core.training.stage2_generate_err import generate_errs

    result = generate_errs(
        detector=detector,
        data=data,
        adapter=adapter,
        teacher=teacher,
        verifier=verifier,
        config=config,
        seed=args.seed,
    )

    # Save
    out_dir = Path(args.output_dir) / "stage2"
    result.save(out_dir)
    manifest_path = registry.write_manifest()
    logger.info(
        "Stage 2 complete: %d accepted, %d rejected saved to %s. Manifest: %s",
        result.num_accepted, result.num_rejected, out_dir, manifest_path,
    )


if __name__ == "__main__":
    main()

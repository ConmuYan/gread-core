"""Stage 1 CLI: Train base detector.

CRITICAL: This CLI must NOT import or use LLM.
"""

from __future__ import annotations

import argparse
import logging

import torch
import yaml

from gread_core.experiment.seed import set_seed


def main(argv: list[str] | None = None) -> None:
    """Stage 1: Train base detector CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GReaD-Core Stage 1: Train base detector"
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
        "--output-dir", type=str, default="artifacts", help="Output dir"
    )
    parser.add_argument(
        "--experiment-id", type=str, default="stage1", help="Experiment ID"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--tensorboard-dir", type=str, default=None,
        help="TensorBoard log dir (disabled if not set)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set seed
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

    # Create detector
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

    # Train
    from gread_core.training.checkpointing import CheckpointManager
    from gread_core.training.stage1_train_detector import train_detector

    ckpt_manager = CheckpointManager(
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        config=config,
    )

    device = torch.device(args.device)
    detector = detector.to(device)
    data = data.to(device)

    writer = None
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.tensorboard_dir)

    train_detector(detector, data, config, ckpt_manager, writer=writer)

    if writer is not None:
        writer.close()

    manifest_path = registry.write_manifest()
    logging.getLogger(__name__).info(
        "Stage 1 complete. Model saved to %s/stage1. Manifest: %s",
        args.output_dir, manifest_path,
    )


if __name__ == "__main__":
    main()

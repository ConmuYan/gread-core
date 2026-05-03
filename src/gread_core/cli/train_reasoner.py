"""Stage 3 CLI: Train reasoner using accepted ERRs.

CRITICAL: Only accepted ERRs are used. Rejected produce zero reasoning loss.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from gread_core.experiment.seed import set_seed


def main(argv: list[str] | None = None) -> None:
    """Stage 3: Train reasoner CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GReaD-Core Stage 3: Train reasoner"
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
        "--detector-checkpoint", type=str, required=True,
        help="Stage 1 checkpoint dir",
    )
    parser.add_argument(
        "--err-dir", type=str, required=True, help="Stage 2 ERR dir"
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts", help="Output dir"
    )
    parser.add_argument(
        "--experiment-id", type=str, default="stage3", help="Experiment ID"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--allow-empty-err", action="store_true",
        help="Allow training with 0 accepted ERRs (testing only)",
    )
    parser.add_argument(
        "--tensorboard-dir", type=str, default=None,
        help="TensorBoard log dir (disabled if not set)",
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

    # Resolve contract_version from YAML if not already in config
    _cv = config.get("verifier", {}).get("contract_version")
    if _cv is None:
        _contract_path = config.get("verifier", {}).get("contract_path")
        if _contract_path:
            _contract_file = Path(_contract_path)
            if _contract_file.exists():
                with open(_contract_file) as cf:
                    _contract_yaml = yaml.safe_load(cf)
                _cv = _contract_yaml.get("contract_version")
                if _cv:
                    config.setdefault("verifier", {})["contract_version"] = _cv

    # Create experiment registry
    from gread_core.experiment.registry import ExperimentRegistry
    registry = ExperimentRegistry(
        experiment_id=args.experiment_id,
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
        contract_version=_cv,
        detector_checkpoint=args.detector_checkpoint,
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

    # Load detector checkpoint
    ckpt_path = Path(args.detector_checkpoint) / "model.pt"
    if ckpt_path.exists():
        detector.load_state_dict(torch.load(ckpt_path, weights_only=True))
        logger.info("Loaded detector from %s", ckpt_path)

    # Load accepted ERRs
    from gread_core.training.stage2_generate_err import Stage2Result

    err_result = Stage2Result.load(args.err_dir)
    logger.info(
        "Loaded %d accepted ERRs from %s",
        err_result.num_accepted, args.err_dir,
    )

    if err_result.num_accepted == 0:
        if args.allow_empty_err:
            logger.warning(
                "0 accepted ERRs loaded (--allow-empty-err set, testing only). "
                "Training will proceed with zero reasoning signal."
            )
        else:
            raise RuntimeError(
                f"0 accepted ERRs loaded from {args.err_dir}. "
                "Stage 3 requires at least 1 accepted ERR to train. "
                "Use --allow-empty-err to override (testing only)."
            )

    # Create reasoner
    from gread_core.models.evidence_encoder import EvidenceEncoder
    from gread_core.models.reasoner import GReaDReasoner

    num_risk_types = 6
    num_evidence_slots = config.get("evidence", {}).get("num_slots", 32)
    rho = config.get("method", {}).get("residual_rho", 0.1)
    signed_evidence_masks = config.get("method", {}).get("signed_evidence_masks", True)

    if rho == 0.0:
        logger.info("ABLATION: residual_rho=0.0 — residual reasoning disabled")
    if not signed_evidence_masks:
        logger.info("ABLATION: signed_evidence_masks=False — using unsigned evidence head")

    evidence_encoder = EvidenceEncoder(
        vocab_size=num_evidence_slots + 10,
        embed_dim=32,
        num_slots=num_evidence_slots,
        output_dim=128,
    )

    reasoner = GReaDReasoner(
        hidden_dim=hidden_channels,
        evidence_encoder=evidence_encoder,
        num_risk_types=num_risk_types,
        num_evidence_slots=num_evidence_slots,
        rho=rho,
        signed_evidence_masks=signed_evidence_masks,
    )

    # Train
    from gread_core.training.checkpointing import CheckpointManager
    from gread_core.training.stage3_train_reasoner import train_reasoner

    contract_version = config.get("verifier", {}).get("contract_version")
    ckpt_manager = CheckpointManager(
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        config=config,
        dataset=args.dataset,
        detector_checkpoint_path=args.detector_checkpoint,
        contract_version=contract_version,
    )

    device = torch.device(args.device)
    reasoner = reasoner.to(device)

    writer = None
    if args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
        writer = SummaryWriter(log_dir=args.tensorboard_dir)  # type: ignore[no-untyped-call]

    train_reasoner(
        reasoner=reasoner,
        detector=detector,
        data=data,
        accepted_errs=err_result.accepted_errs,
        config=config,
        checkpoint_manager=ckpt_manager,
        device=device,
        writer=writer,
    )

    if writer is not None:
        writer.close()  # type: ignore[no-untyped-call]

    manifest_path = registry.write_manifest()
    logger.info(
        "Stage 3 complete. Reasoner saved to %s/stage3. Manifest: %s",
        args.output_dir, manifest_path,
    )


if __name__ == "__main__":
    main()

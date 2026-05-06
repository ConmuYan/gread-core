"""Evaluation CLI: run all evaluation metrics on a trained model.

Usage:
    python -m gread_core.cli.evaluate --checkpoint path --config path --output artifacts/metrics/
    python -m gread_core.cli.evaluate --checkpoint path --config path --dataset yelpchi \
        --detector gcn --detector-checkpoint path --output artifacts/metrics/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from gread_core.detectors.factory import (
    ALL_DETECTOR_TYPES,
    create_detector,
    get_detector_embedding_dim,
)
from gread_core.evaluation.ablation import run_ablation
from gread_core.evaluation.cec import compute_tri_cec
from gread_core.evaluation.detection import compute_all_detection_metrics
from gread_core.evaluation.non_redundancy import compute_non_redundancy
from gread_core.evaluation.reasoning import compute_all_reasoning_metrics
from gread_core.experiment.seed import set_seed
from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.reasoner import GReaDReasoner
from gread_core.schemas.err import RiskType

_RISK_TYPES: list[str] = list(RiskType.__args__)  # type: ignore[attr-defined]


def _load_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> GReaDReasoner:
    """Load a GReaDReasoner from checkpoint."""
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)
    num_evidence_slots = config.get("evidence", {}).get("num_slots", 32)
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


def _load_detector(
    config: dict[str, Any],
    detector_type: str,
    checkpoint_path: Path,
    in_channels: int,
    hidden_channels: int,
) -> torch.nn.Module:
    """Load a base detector from checkpoint."""
    detector = create_detector(
        detector_type,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        config=config,
    )

    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        detector.load_state_dict(torch.load(model_path, weights_only=True))

    return detector


def _generate_synthetic_data(
    n: int,
    num_slots: int,
    hidden_dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:  # noqa: E501
    """Generate synthetic evaluation data for demo/testing."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, size=n).astype(np.float64)
    y_score = rng.uniform(0, 1, size=n)
    y_pred = (y_score > 0.5).astype(np.float64)
    types_encoded = np.eye(6)[rng.randint(0, 6, size=n)]

    z_v = torch.randn(n, hidden_dim)
    base_logit = torch.randn(n)
    evidence_token_ids = torch.randint(0, num_slots, (n, num_slots))

    evidence_features = torch.sigmoid(torch.randn(n, num_slots)).numpy()

    return (
        y_true, y_score, y_pred, types_encoded,
        z_v, base_logit, evidence_token_ids, evidence_features,
    )


def resolve_evaluation_mode(args: Any) -> str:
    """Resolve evaluation mode and fail closed for formal runs."""
    if args.synthetic:
        return "synthetic"

    missing = [
        flag
        for flag, value in (
            ("--dataset", args.dataset),
            ("--detector", args.detector),
            ("--detector-checkpoint", args.detector_checkpoint),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Real evaluation requires --dataset, --detector, "
            "--detector-checkpoint, and --err-dir. "
            f"Missing: {', '.join(missing)}. "
            "Pass --synthetic explicitly for legacy synthetic evaluation."
        )
    if args.err_dir is None:
        raise ValueError("Real evaluation requires --err-dir for accepted ERR references.")
    return "real"


def _run_real_inference(
    config: dict[str, Any],
    dataset: str,
    detector_type: str,
    detector_checkpoint: Path,
    reasoner: GReaDReasoner,
    device: torch.device,
    seed: int,
    data_root: str | None,
    logger: logging.Logger,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, list[int],
]:
    """Run real inference on test nodes and return evaluation data."""
    from gread_core.data.loaders import load_graph_dataset
    from gread_core.training.stage2_generate_err import _strip_masks

    # Load dataset
    data = load_graph_dataset(dataset, root=data_root, seed=seed, config=config)
    in_channels = data.x.shape[1]
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)

    # Load detector
    detector = _load_detector(
        config, detector_type, detector_checkpoint, in_channels, hidden_channels
    )
    detector.to(device)
    detector.eval()

    # Get detector outputs for ALL nodes
    no_mask_data = _strip_masks(data)
    no_mask_data = no_mask_data.to(device)
    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(no_mask_data)  # type: ignore[operator]
        base_logits = logits.cpu()
        embeddings_cpu = embeddings.cpu()

    # Get predictions via reasoner
    from gread_core.adapters.factory import create_evidence_adapter
    adapter_thresholds = config.get("adapter", {}).get("thresholds", {})
    adapter = create_evidence_adapter(
        detector_type=detector_type,
        detector=detector,
        graph=data,
        logits=logits.cpu(),
        embeddings=embeddings.cpu(),
        thresholds=adapter_thresholds,
        strict_detector_signal=dataset != "tiny",
    )

    # Build evidence for test nodes
    labels = data.y.cpu()
    test_mask = data.test_mask
    if test_mask is None:
        # If no test mask, use all nodes
        test_nodes = list(range(data.x.shape[0]))
    else:
        test_nodes = test_mask.nonzero(as_tuple=True)[0].tolist()

    logger.info("Running inference on %d test nodes", len(test_nodes))

    # Extract MEPs for test nodes
    meps = adapter.extract(test_nodes)

    # Encode evidence using canonical slot vocabulary
    from gread_core.schemas.risk_taxonomy import encode_evidence_slots
    num_slots = config.get("evidence", {}).get("num_slots", 32)

    evidence_ids_list = []
    for mep in meps:
        reasoning = mep.reasoning
        field_names = [
            "uncertainty_level", "degree_level", "neighbor_consistency",
            "feature_neighbor_discrepancy", "detector_signal",
            "detector_signal_strength", "counter_signal",
        ]
        present = [f for f in field_names if getattr(reasoning, f, None) is not None]
        ids = encode_evidence_slots(present, num_slots)
        evidence_ids_list.append(ids)

    evidence_ids = torch.tensor(evidence_ids_list, dtype=torch.long)

    # Reasoner forward
    z_v = embeddings_cpu[test_nodes]
    base_logit = base_logits[test_nodes]

    reasoner.eval()
    with torch.no_grad():
        outputs = reasoner.forward(
            z_v=z_v.to(device),
            base_logit=base_logit.to(device),
            evidence_token_ids=evidence_ids.to(device),
        )

    final_logit = outputs["final_logit"].cpu()
    y_score = torch.sigmoid(final_logit).numpy()
    y_pred = (y_score > 0.5).astype(np.float64)
    y_true = labels[test_nodes].numpy().astype(np.float64)

    # Get type predictions for non-redundancy
    type_logits = outputs["type_logits"].cpu()
    types_encoded = torch.softmax(type_logits, dim=-1).numpy()

    # Get evidence mask features for non-redundancy
    pos_mask = torch.sigmoid(outputs["pos_mask_logits"].cpu())
    neg_mask = torch.sigmoid(outputs["neg_mask_logits"].cpu())
    evidence_features = torch.cat([pos_mask, neg_mask], dim=-1).numpy()

    logger.info("Real inference complete: %d test nodes, fraud_ratio=%.2f",
                len(test_nodes), y_true.mean())

    return (
        y_true, y_score, y_pred, types_encoded,
        z_v, base_logit, evidence_ids, evidence_features, test_nodes,
    )


def main(argv: list[str] | None = None) -> None:
    """Evaluation CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GReaD-Core Evaluation Suite"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Reasoner checkpoint directory"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config path"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/metrics/", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Real dataset root; overrides config data.root and GREAD_DATA_ROOT",
    )
    parser.add_argument(
        "--experiment-id", type=str, default="eval", help="Experiment ID"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (required for real evaluation)"
    )
    parser.add_argument(
        "--detector", type=str, default=None,
        choices=ALL_DETECTOR_TYPES,
        help="Detector type (required for real evaluation)"
    )
    parser.add_argument(
        "--detector-checkpoint", type=str, default=None,
        help="Stage 1 detector checkpoint (required for real evaluation)"
    )
    parser.add_argument(
        "--err-dir", type=str, default=None,
        help="Stage 2 ERR output directory (accepted_errs.json) for reference reasoning metrics"
    )
    parser.add_argument(
        "--skip-ablation", action="store_true", help="Skip ablation experiments"
    )
    parser.add_argument(
        "--skip-cec", action="store_true", help="Skip CEC evaluation"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (legacy mode, for testing only)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.detector is not None:
        config.setdefault("detector", {})["type"] = args.detector

    set_seed(args.seed)
    mode = resolve_evaluation_mode(args)
    use_real = mode == "real"

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
        dataset=args.dataset or "synthetic",
        seed=args.seed,
        output_dir=args.output,
        contract_version=_cv,
        detector_checkpoint=args.detector_checkpoint,
    )

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    num_slots = config.get("evidence", {}).get("num_slots", 32)
    hidden_dim = config.get("detector", {}).get("hidden_channels", 64)

    if use_real:
        assert args.dataset is not None
        assert args.detector is not None
        assert args.detector_checkpoint is not None
        logger.info("Running REAL evaluation on dataset=%s detector=%s",
                     args.dataset, args.detector)
        reasoner = _load_model(checkpoint_path, config, device)
        (
            y_true, y_score, y_pred, types_encoded,
            z_v, base_logit, evidence_ids, evidence_features, test_nodes,
        ) = _run_real_inference(
            config, args.dataset, args.detector,
            Path(args.detector_checkpoint), reasoner, device, args.seed,
            args.data_root, logger,
        )
    else:
        logger.info("Running SYNTHETIC evaluation (explicit legacy mode)")
        y_true, y_score, y_pred, types_encoded, z_v, base_logit, evidence_ids, evidence_features = (
            _generate_synthetic_data(200, num_slots, hidden_dim, args.seed)
        )
        test_nodes = list(range(len(y_true)))

    # Detection metrics
    logger.info("Computing detection metrics")
    detection_metrics = compute_all_detection_metrics(y_true, y_score, y_pred)
    logger.info("Detection: AUC=%.4f AUPRC=%.4f F1=%.4f",
                detection_metrics["auc"], detection_metrics["auprc"],
                detection_metrics["f1"])

    # Reasoning metrics
    logger.info("Computing reasoning metrics")
    n = len(y_true)

    if use_real:
        slot_names = config.get("evidence", {}).get(
            "evidence_slot_names",
            [f"slot_{i}" for i in range(num_slots)],
        )

        def _evidence_ids_to_names(ids_tensor: torch.Tensor) -> list[str]:
            """Convert evidence_ids row (shape [num_slots]) to list of slot names."""
            names: list[str] = []
            for idx in range(ids_tensor.shape[0]):
                val = ids_tensor[idx].item()
                if 1 <= val <= len(slot_names):
                    names.append(slot_names[val - 1])
            return names

        def _type_idx_to_name(probs: np.ndarray) -> str:
            """Convert softmax type probabilities to risk type name."""
            idx = int(np.argmax(probs))
            if idx < len(_RISK_TYPES):
                return _RISK_TYPES[idx]
            return _RISK_TYPES[0]

        # Build predictions from inference results
        predictions = []
        for i in range(n):
            predictions.append({
                "accepted": bool(y_pred[i]),
                "evidence": _evidence_ids_to_names(evidence_ids[i]),
                "risk_type": _type_idx_to_name(types_encoded[i]),
            })

        # Build references from Stage 2 accepted ERRs
        references = []
        err_dir = args.err_dir
        err_lookup: dict[int, dict[str, Any]] = {}
        if err_dir is not None:
            err_path = Path(err_dir) / "accepted_errs.json"
            if err_path.exists():
                with open(err_path) as f:
                    accepted_errs = json.load(f)
                for err in accepted_errs:
                    node_idx = err.get("node_idx")
                    if node_idx is not None:
                        err_lookup[int(node_idx)] = err
                logger.info("Loaded %d accepted ERRs from %s", len(err_lookup), err_path)
            else:
                logger.warning("accepted_errs.json not found at %s", err_path)
        else:
            logger.warning("No --err-dir provided; references will use empty evidence")

        for i in range(n):
            # Use global node index for ERR lookup, not local loop index
            global_node_idx = test_nodes[i] if i < len(test_nodes) else i
            matched = err_lookup.get(global_node_idx)
            if matched and "err" in matched:
                err_data = matched["err"]
                references.append({
                    "accepted": bool(y_true[i]),
                    "evidence": err_data.get("supporting_evidence", []),
                    "risk_type": err_data.get("risk_type", _RISK_TYPES[0]),
                })
            else:
                references.append({
                    "accepted": bool(y_true[i]),
                    "evidence": [],
                    "risk_type": _RISK_TYPES[0],
                })
    else:
        logger.warning("Using synthetic/dummy reasoning metrics — not meaningful for paper")
        predictions = [
            {"accepted": bool(y_pred[i]), "evidence": [f"e{j}" for j in range(3)],
             "risk_type": "camouflage_neighbor"}
            for i in range(n)
        ]
        references = [
            {"accepted": bool(y_true[i]), "evidence": [f"e{j}" for j in range(3)],
             "risk_type": "camouflage_neighbor"}
            for i in range(n)
        ]

    reasoning_metrics = compute_all_reasoning_metrics(predictions, references)
    logger.info("Reasoning: acceptance=%.4f evidence_f1=%.4f type_acc=%.4f",
                reasoning_metrics["acceptance_rate"],
                reasoning_metrics["evidence_f1"],
                reasoning_metrics["risk_type_accuracy"])

    # Non-redundancy
    logger.info("Computing non-redundancy metrics")
    non_red = compute_non_redundancy(y_score, types_encoded, evidence_features, y_true)
    logger.info("Non-redundancy: AUC improvements computed")

    all_metrics: dict[str, Any] = {
        "evaluation_mode": mode,
        "dataset": args.dataset or "synthetic",
        "detector": args.detector or "synthetic",
        "detection": detection_metrics,
        "reasoning": reasoning_metrics,
        "non_redundancy": {
            "auc_improvement": non_red["auc_improvement"],
            "auprc_improvement": non_red["auprc_improvement"],
            "models": non_red["models"],
        },
    }

    # Load model for ablation/CEC if checkpoint exists
    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        if not use_real:
            reasoner = _load_model(checkpoint_path, config, device)

        z_v_dev = z_v.to(device)
        base_logit_dev = base_logit.to(device)
        evidence_ids_dev = evidence_ids.to(device)

        if not args.skip_ablation:
            logger.info("Running ablation experiments")
            ablation_results = run_ablation(
                reasoner, z_v_dev, base_logit_dev, evidence_ids_dev
            )
            ablation_summary = {}
            for name, outputs in ablation_results.items():
                ablation_summary[name] = {
                    k: v.mean().item() if isinstance(v, torch.Tensor) else v
                    for k, v in outputs.items()
                }
            all_metrics["ablation"] = ablation_summary

        if not args.skip_cec:
            logger.info("Computing tri-CEC metrics")
            from gread_core.schemas.evidence import (
                CalibrationChannel,
                MinimalEvidencePackage,
                ReasoningChannel,
            )

            if use_real:
                assert args.dataset is not None
                assert args.detector is not None
                assert args.detector_checkpoint is not None
                # Use real MEPs from the dataset
                from gread_core.adapters.factory import create_evidence_adapter
                from gread_core.data.loaders import load_graph_dataset
                from gread_core.training.stage2_generate_err import _strip_masks

                data = load_graph_dataset(
                    args.dataset, root=args.data_root, seed=args.seed, config=config
                )
                in_ch = data.x.shape[1]
                h_ch = config.get("detector", {}).get("hidden_channels", 64)
                det = _load_detector(
                    config, args.detector, Path(args.detector_checkpoint), in_ch, h_ch
                )
                det.to(device)
                det.eval()

                no_mask = _strip_masks(data).to(device)
                with torch.no_grad():
                    logits_emb, embeddings_emb = det.forward_with_embedding(no_mask)  # type: ignore[operator]

                adapter_thresholds = config.get("adapter", {}).get("thresholds", {})
                adapter = create_evidence_adapter(
                    detector_type=args.detector,
                    detector=det,
                    graph=data,
                    logits=logits_emb.cpu(),
                    embeddings=embeddings_emb.cpu(),
                    thresholds=adapter_thresholds,
                    strict_detector_signal=args.dataset != "tiny",
                )
                test_m = data.test_mask
                if test_m is None:
                    tn = list(range(min(10, data.x.shape[0])))
                else:
                    tn = test_m.nonzero(as_tuple=True)[0][:10].tolist()

                cec_meps = adapter.extract(tn)

                # Use canonical EVIDENCE_SLOT_TO_INDEX for consistent tokenization
                from gread_core.schemas.risk_taxonomy import EVIDENCE_SLOT_TO_INDEX
                slot_to_id = dict(EVIDENCE_SLOT_TO_INDEX)

                cec_results = compute_tri_cec(
                    reasoner, cec_meps, slot_to_id, num_slots,
                    z_v_dev[:len(cec_meps)], base_logit_dev[:len(cec_meps)],
                )
                all_metrics["tri_cec_node_ids"] = tn
            else:
                # Synthetic dummy MEPs for testing
                slot_to_id = {f"slot_{i}": i for i in range(num_slots)}
                dummy_meps = []
                for _i in range(min(10, n)):
                    dummy_meps.append(MinimalEvidencePackage(
                        node_id="n0",
                        detector_name="synthetic",
                        calibration=CalibrationChannel(prediction_score=0.5, uncertainty=0.5),
                        reasoning=ReasoningChannel(
                            uncertainty_level="medium",
                            degree_level="high",
                            neighbor_consistency="low",
                            feature_neighbor_discrepancy="high",
                            detector_signal="high_frequency_response_high",
                            detector_signal_strength="strong",
                            counter_signal="benign_neighbor_signal_low",
                            allowed_support_ids=["degree_level", "neighbor_consistency",
                                                 "feature_neighbor_discrepancy", "detector_signal",
                                                 "detector_signal_strength"],
                            allowed_counter_ids=["counter_signal", "uncertainty_level"],
                        ),
                    ))
                cec_results = compute_tri_cec(
                    reasoner, dummy_meps, slot_to_id, num_slots,
                    z_v_dev[:len(dummy_meps)], base_logit_dev[:len(dummy_meps)],
                )
            all_metrics["tri_cec"] = cec_results

    # Save results
    manifest_path = registry.write_manifest()
    logger.info("Manifest written to %s", manifest_path)
    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()

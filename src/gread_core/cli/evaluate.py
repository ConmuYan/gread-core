"""Evaluation CLI: run all evaluation metrics on a trained model.

Usage:
    python -m gread_core.cli.evaluate --checkpoint path --config path --output artifacts/metrics/
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

from gread_core.evaluation.ablation import run_ablation
from gread_core.evaluation.cec import compute_tri_cec
from gread_core.evaluation.detection import compute_all_detection_metrics
from gread_core.evaluation.non_redundancy import compute_non_redundancy
from gread_core.evaluation.reasoning import compute_all_reasoning_metrics
from gread_core.experiment.seed import set_seed
from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.reasoner import GReaDReasoner


def _load_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> GReaDReasoner:
    """Load a GReaDReasoner from checkpoint."""
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)
    num_evidence_slots = config.get("evidence", {}).get("num_slots", 32)
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


def _generate_synthetic_data(
    n: int,
    num_slots: int,
    hidden_dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: E501
    """Generate synthetic evaluation data for demo/testing."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, size=n).astype(np.float64)
    y_score = rng.uniform(0, 1, size=n)
    y_pred = (y_score > 0.5).astype(np.float64)
    types_encoded = np.eye(6)[rng.randint(0, 6, size=n)]

    z_v = torch.randn(n, hidden_dim)
    base_logit = torch.randn(n)
    evidence_token_ids = torch.randint(0, num_slots, (n, num_slots))

    return y_true, y_score, y_pred, types_encoded, z_v, base_logit, evidence_token_ids


def main(argv: list[str] | None = None) -> None:
    """Evaluation CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GReaD-Core Evaluation Suite"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint directory"
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
        "--experiment-id", type=str, default="eval", help="Experiment ID"
    )
    parser.add_argument(
        "--dataset", type=str, default="unknown", help="Dataset name"
    )
    parser.add_argument(
        "--skip-ablation", action="store_true", help="Skip ablation experiments"
    )
    parser.add_argument(
        "--skip-cec", action="store_true", help="Skip CEC evaluation"
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

    set_seed(args.seed)

    # Create experiment registry
    from gread_core.experiment.registry import ExperimentRegistry
    registry = ExperimentRegistry(
        experiment_id=args.experiment_id,
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        seed=args.seed,
        output_dir=args.output,
    )

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)

    num_slots = config.get("evidence", {}).get("num_slots", 32)
    hidden_dim = config.get("detector", {}).get("hidden_channels", 64)

    # Generate synthetic data for demo
    logger.info("Generating synthetic evaluation data")
    y_true, y_score, y_pred, types_encoded, z_v, base_logit, evidence_ids = (
        _generate_synthetic_data(200, num_slots, hidden_dim, args.seed)
    )

    # Detection metrics
    logger.info("Computing detection metrics")
    detection_metrics = compute_all_detection_metrics(y_true, y_score, y_pred)
    logger.info("Detection: AUC=%.4f AUPRC=%.4f F1=%.4f",
                detection_metrics["auc"], detection_metrics["auprc"],
                detection_metrics["f1"])

    # Reasoning metrics (synthetic)
    logger.info("Computing reasoning metrics")
    n = len(y_true)
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
    evidence_features = torch.sigmoid(
        torch.randn(n, num_slots)
    ).numpy()
    non_red = compute_non_redundancy(y_score, types_encoded, evidence_features, y_true)
    logger.info("Non-redundancy: AUC improvements computed")

    all_metrics: dict[str, Any] = {
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
            from gread_core.schemas.evidence import MinimalEvidencePackage
            # Create dummy MEPs for CEC (synthetic)
            slot_to_id = {f"slot_{i}": i for i in range(num_slots)}
            dummy_meps = []
            from gread_core.schemas.evidence import CalibrationChannel, ReasoningChannel
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

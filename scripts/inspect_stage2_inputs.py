"""Inspect Stage 2 adapter outputs before offline LLM ERR generation.

This is a pre-flight gate for formal runs. It loads a trained detector,
extracts score-blind MEPs for the selected trace nodes, and writes the
calibration/evidence/prompt payload channels separately so leakage can be
audited before any local LLM generation is launched.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml

from gread_core.adapters.diagnostics import build_native_evidence_distribution_report
from gread_core.adapters.factory import create_evidence_adapter
from gread_core.cli.generate_err import resolve_stage2_runtime
from gread_core.data.loaders import load_graph_dataset
from gread_core.detectors.factory import create_detector
from gread_core.experiment.seed import set_seed
from gread_core.llm.prompt_builder import PromptBuilder
from gread_core.tracing.buckets import assign_buckets
from gread_core.tracing.selector import TraceSelector
from gread_core.training.stage2_generate_err import (
    _compute_uncertainties,
    _resolve_trace_split_mask,
    _strip_masks,
)

FORBIDDEN_PAYLOAD_KEYS = {
    "prediction_score",
    "fraud_score",
    "base_score",
    "probability",
    "probability_score",
    "logit",
    "rank",
    "confidence",
    "predicted_label",
}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _flatten_keys(obj: Any, prefix: str = "") -> list[str]:
    if isinstance(obj, dict):
        keys: list[str] = []
        for key, value in obj.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            keys.append(full)
            keys.extend(_flatten_keys(value, full))
        return keys
    if isinstance(obj, list):
        keys = []
        for idx, value in enumerate(obj):
            keys.extend(_flatten_keys(value, f"{prefix}[{idx}]"))
        return keys
    return []


def _payload_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload", row)
        key_paths = _flatten_keys(payload)
        raw = json.dumps(payload, sort_keys=True).lower()
        found_keys = [
            path for path in key_paths
            if path.split(".")[-1].lower() in FORBIDDEN_PAYLOAD_KEYS
        ]
        found_tokens = [
            token for token in sorted(FORBIDDEN_PAYLOAD_KEYS)
            if token in raw
        ]
        if found_keys or found_tokens:
            violations.append({
                "node_id": row.get("node_id"),
                "keys": found_keys,
                "tokens": found_tokens,
            })
    return {
        "ok": not violations,
        "num_checked": len(rows),
        "num_violations": len(violations),
        "violations": violations[:20],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--trace-budget", type=int, default=None)
    parser.add_argument("--trace-split", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("inspect_stage2_inputs")

    config = _load_yaml(args.config)
    if args.trace_budget is not None:
        config.setdefault("trace_selection", {})["total_budget"] = args.trace_budget
    if args.trace_split is not None:
        config.setdefault("stage2", {})["trace_split"] = args.trace_split

    runtime = resolve_stage2_runtime(config, dataset=args.dataset)
    set_seed(args.seed)
    device = torch.device(args.device)

    data = load_graph_dataset(
        args.dataset,
        root=args.data_root,
        seed=args.seed,
        config=config,
    ).to(device)

    detector = create_detector(
        args.detector,
        in_channels=data.x.shape[1],
        hidden_channels=config.get("detector", {}).get("hidden_channels", 64),
        config=config,
    )
    checkpoint = Path(args.checkpoint)
    model_path = checkpoint / "model.pt"
    detector.load_state_dict(torch.load(model_path, weights_only=True))
    detector = detector.to(device)
    detector.eval()

    no_mask_data = _strip_masks(data)
    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(no_mask_data)
        scores = torch.sigmoid(logits).cpu()
        uncertainties = _compute_uncertainties(scores, embeddings).cpu()

    labels = data.y.cpu() if hasattr(data, "y") else None
    trace_split = config.get("stage2", {}).get("trace_split", "all")
    trace_mask = _resolve_trace_split_mask(data, trace_split)
    bucket_assignments = assign_buckets(scores, uncertainties, labels)
    if trace_mask is not None:
        bucket_assignments = [
            assignment if bool(trace_mask[i]) else None
            for i, assignment in enumerate(bucket_assignments)
        ]

    adapter = create_evidence_adapter(
        detector_type=args.detector,
        detector=detector,
        graph=data,
        logits=logits,
        embeddings=embeddings,
        thresholds=config.get("adapter", {}).get("thresholds", {}),
        strict_detector_signal=args.dataset != "tiny",
    )

    bucket_nodes: dict[str, list[int]] = {
        "uncertain": [],
        "high_conf_fraud": [],
        "high_conf_benign": [],
    }
    for idx, assignment in enumerate(bucket_assignments):
        if assignment is not None:
            bucket_nodes[assignment].append(idx)
    all_candidates = [idx for nodes in bucket_nodes.values() for idx in nodes]
    candidate_meps_list = adapter.extract(all_candidates)
    candidate_meps = dict(zip(all_candidates, candidate_meps_list, strict=True))

    class _MEPProxy:
        def __getitem__(self, idx: int) -> Any:
            return candidate_meps[idx]
        def __len__(self) -> int:
            return len(candidate_meps)

    selection = TraceSelector(config, args.seed).select(
        scores,
        uncertainties,
        labels,
        _MEPProxy(),  # type: ignore[arg-type]
        bucket_assignments=bucket_assignments,
    )
    selected_meps = [candidate_meps[idx] for idx in selection.node_ids]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    calibration_rows = []
    evidence_rows = []
    mep_rows = []
    prompt_rows = []
    prompt_builder = PromptBuilder(score_blind=config.get("method", {}).get("score_blind", True))
    for node_idx, bucket, div_score, mep in zip(
        selection.node_ids,
        selection.bucket_labels,
        selection.diversity_scores,
        selected_meps,
        strict=True,
    ):
        calibration_rows.append({
            "node_id": mep.node_id,
            "node_idx": node_idx,
            "bucket": bucket,
            "prediction_score": mep.calibration.prediction_score,
            "uncertainty": mep.calibration.uncertainty,
        })
        evidence_rows.append({
            "node_id": mep.node_id,
            "node_idx": node_idx,
            "bucket": bucket,
            "diversity_score": div_score,
            "source": f"{args.detector}_adapter",
            "payload": mep.reasoning.model_dump(),
        })
        teacher_payload = mep.to_teacher_payload()
        mep_rows.append({
            "node_id": mep.node_id,
            "node_idx": node_idx,
            "bucket": bucket,
            "payload": teacher_payload,
        })
        prompt_rows.append({
            "node_id": mep.node_id,
            "node_idx": node_idx,
            "prompt": prompt_builder.build(teacher_payload),
        })

    _write_jsonl(out / "calibration.jsonl", calibration_rows)
    _write_jsonl(out / "native_evidence_candidates.jsonl", evidence_rows)
    _write_jsonl(out / "mep_inputs.jsonl", mep_rows)
    _write_jsonl(out / "prompt_inputs.jsonl", prompt_rows)

    reasonings = [mep.reasoning for mep in selected_meps]
    adapter_report = {
        "dataset": args.dataset,
        "detector": args.detector,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "trace_split": trace_split,
        "trace_budget": config.get("trace_selection", {}).get("total_budget"),
        "llm_backend": runtime.llm_backend,
        "llm_model": runtime.llm_model,
        "llm_model_path": runtime.model_path,
        "num_nodes": int(scores.shape[0]),
        "eligible_by_bucket": {key: len(value) for key, value in bucket_nodes.items()},
        "selected_by_bucket": dict(Counter(selection.bucket_labels)),
        "num_selected": len(selection.node_ids),
        "detector_signal_counts": dict(Counter(r.detector_signal for r in reasonings)),
        "detector_signal_strength_counts": dict(
            Counter(r.detector_signal_strength for r in reasonings)
        ),
        "uncertainty_level_counts": dict(Counter(r.uncertainty_level for r in reasonings)),
        "supports_detector_signal": adapter.supports_detector_signal(),
    }
    with open(out / "adapter_report.json", "w") as f:
        json.dump(adapter_report, f, indent=2, sort_keys=True)

    native_evidence_report = build_native_evidence_distribution_report(
        adapter=adapter,
        meps=selected_meps,
        node_ids=selection.node_ids,
        source_metadata={
            "dataset": args.dataset,
            "detector": args.detector,
            "seed": args.seed,
            "checkpoint": str(checkpoint),
            "trace_split": trace_split,
            "trace_budget": config.get("trace_selection", {}).get("total_budget"),
            "llm_backend": runtime.llm_backend,
            "llm_model": runtime.llm_model,
            "llm_model_path": runtime.model_path,
            "num_graph_nodes": int(scores.shape[0]),
            "eligible_by_bucket": {
                key: len(value) for key, value in bucket_nodes.items()
            },
            "selected_by_bucket": dict(Counter(selection.bucket_labels)),
        },
    )
    with open(out / "native_evidence_distribution_report.json", "w") as f:
        json.dump(native_evidence_report, f, indent=2, sort_keys=True)

    prompt_leaks = []
    for row in prompt_rows:
        prompt = row["prompt"].lower()
        raw_score = str(next(
            c["prediction_score"] for c in calibration_rows
            if c["node_id"] == row["node_id"]
        ))
        found = []
        # The prompt template may contain natural-language prohibitions such as
        # "Do not mention probability"; those are guardrails, not leakage.
        # Fail only on schema-style keys or the raw calibration value.
        for token in ("prediction_score", "fraud_score", "base_score", "probability_score"):
            if token in prompt:
                found.append(token)
        if raw_score in prompt:
            found.append("raw_prediction_score_value")
        if found:
            prompt_leaks.append({"node_id": row["node_id"], "tokens": found})

    leakage_report = {
        "calibration_channel_contains_prediction_score": True,
        "native_evidence_payload": _payload_leakage(evidence_rows),
        "mep_teacher_payload": _payload_leakage(mep_rows),
        "prompt_text": {
            "ok": not prompt_leaks,
            "num_checked": len(prompt_rows),
            "num_violations": len(prompt_leaks),
            "violations": prompt_leaks[:20],
        },
    }
    leakage_report["ok"] = (
        leakage_report["native_evidence_payload"]["ok"]
        and leakage_report["mep_teacher_payload"]["ok"]
        and leakage_report["prompt_text"]["ok"]
    )
    with open(out / "leakage_report.json", "w") as f:
        json.dump(leakage_report, f, indent=2, sort_keys=True)

    logger.info("Wrote Stage 2 pre-flight artifacts to %s", out)
    if not leakage_report["ok"]:
        raise SystemExit("Leakage report failed; inspect leakage_report.json")


if __name__ == "__main__":
    main()

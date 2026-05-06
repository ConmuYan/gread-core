"""Score-blind diagnostics for detector-native adapter evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import ceil, floor, isclose
from typing import Any

from gread_core.schemas.evidence import MinimalEvidencePackage

FORBIDDEN_SCORE_KEYS: frozenset[str] = frozenset(
    {
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
)


def build_native_evidence_distribution_report(
    *,
    adapter: Any,
    meps: list[MinimalEvidencePackage],
    node_ids: list[int],
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detector-agnostic, score-blind native evidence report."""
    if len(meps) != len(node_ids):
        msg = "meps and node_ids must have the same length"
        raise ValueError(msg)

    reasonings = [mep.reasoning for mep in meps]
    detector_signals = [r.detector_signal for r in reasonings]
    strengths = [r.detector_signal_strength for r in reasonings]
    counter_signals = [r.counter_signal for r in reasonings]
    raw_metric = _raw_metric_report(adapter, node_ids)

    distributions = {
        "detector_signal": _distribution(detector_signals),
        "detector_signal_strength": _distribution(strengths),
        "counter_signal": _distribution(counter_signals),
    }
    audit = _score_blind_audit(meps, raw_metric)

    metadata = dict(source_metadata or {})
    metadata.update(
        {
            "adapter_class": adapter.__class__.__name__,
            "detector_name": getattr(adapter, "detector_name", "unknown"),
            "supports_detector_signal": bool(adapter.supports_detector_signal()),
            "num_records": len(meps),
            "node_scope": "provided_node_ids",
            "report_schema_version": 1,
        }
    )

    report = {
        "source_metadata": metadata,
        "raw_metric": raw_metric,
        "distributions": distributions,
        "degeneracy_flags": _degeneracy_flags(
            distributions=distributions,
            raw_metric=raw_metric,
            supports_detector_signal=metadata["supports_detector_signal"],
        ),
        "score_blind_audit": audit,
    }
    report["ok"] = audit["ok"]
    return report


def _distribution(values: Sequence[str]) -> dict[str, Any]:
    counts = dict(sorted(Counter(values).items()))
    total = len(values)
    proportions = {
        key: (value / total if total else 0.0)
        for key, value in counts.items()
    }
    dominant_value = None
    dominant_ratio = 0.0
    if counts:
        dominant_value, dominant_count = max(
            counts.items(), key=lambda item: (item[1], item[0])
        )
        dominant_ratio = dominant_count / total
    return {
        "count": total,
        "num_unique": len(counts),
        "counts": counts,
        "proportions": proportions,
        "dominant_value": dominant_value,
        "dominant_ratio": dominant_ratio,
    }


def _raw_metric_report(adapter: Any, node_ids: list[int]) -> dict[str, Any]:
    values, metric_name, source = _extract_raw_metric_values(adapter, node_ids)
    available = bool(values)
    return {
        "available": available,
        "metric_name": metric_name,
        "source": source,
        "count": len(values),
        "quantiles": _quantiles(values) if available else {},
    }


def _extract_raw_metric_values(
    adapter: Any, node_ids: list[int]
) -> tuple[list[float], str | None, str]:
    if hasattr(adapter, "_spectral_responses"):
        values = _spectral_response_values(adapter, node_ids)
        if values:
            return values, "spectral_energy_ratio", "adapter.spectral_responses"

    if hasattr(adapter, "_filter_weights"):
        values = _filter_weight_values(adapter, node_ids)
        if values:
            return values, "filter_disagreement", "adapter.filter_weights"

    if hasattr(adapter, "_feature_importance"):
        values = _feature_importance_values(adapter, node_ids)
        if values:
            return values, "feature_importance_variance", "adapter.feature_importance"

    if hasattr(adapter, "_native_values"):
        values = _native_values(adapter, node_ids)
        if values:
            family = getattr(adapter, "_signal_family", "native")
            return values, f"{family}_native_metric", "adapter.native_values"

    if hasattr(adapter, "_embeddings"):
        values = _embedding_values(adapter, node_ids)
        if values:
            return values, "embedding_neighbor_cosine_distance", "adapter.embeddings"

    return [], None, "unavailable"


def _spectral_response_values(adapter: Any, node_ids: list[int]) -> list[float]:
    from gread_core.adapters.bwgnn_adapter import _spectral_energy_ratio

    spectral = getattr(adapter, "_spectral_responses", None)
    if spectral is None or spectral.numel() == 0:
        return []
    values = []
    for node_id in node_ids:
        if node_id < spectral.shape[0]:
            values.append(float(_spectral_energy_ratio(spectral[node_id])))
    return values


def _filter_weight_values(adapter: Any, node_ids: list[int]) -> list[float]:
    from gread_core.adapters.caregnn_adapter import _filter_disagreement

    weights = getattr(adapter, "_filter_weights", None)
    if weights is None:
        return []
    values = []
    for node_id in node_ids:
        if node_id in weights:
            values.append(float(_filter_disagreement(weights[node_id])))
    return values


def _feature_importance_values(adapter: Any, node_ids: list[int]) -> list[float]:
    importance = getattr(adapter, "_feature_importance", None)
    if importance is None or importance.numel() == 0:
        return []
    variance = float(importance.detach().float().var().item())
    return [variance for _ in node_ids]


def _native_values(adapter: Any, node_ids: list[int]) -> list[float]:
    from gread_core.adapters.pyg_gnn_adapter import _native_metric

    native = getattr(adapter, "_native_values", None)
    signal_family = getattr(adapter, "_signal_family", "native")
    if (
        native is None
        or native.numel() == 0
        or signal_family == "embedding"
    ):
        return []
    values = []
    for node_id in node_ids:
        if node_id < native.shape[0]:
            values.append(float(_native_metric(signal_family, native[node_id])))
    return values


def _embedding_values(adapter: Any, node_ids: list[int]) -> list[float]:
    from gread_core.adapters.pyg_gnn_adapter import _embedding_cosine_distance

    embeddings = getattr(adapter, "_embeddings", None)
    graph = getattr(adapter, "_graph", None)
    if (
        embeddings is None
        or embeddings.numel() == 0
        or graph is None
        or not hasattr(graph, "edge_index")
    ):
        return []

    values = []
    edge_index = graph.edge_index
    for node_id in node_ids:
        if node_id >= embeddings.shape[0]:
            continue
        mask = edge_index[0] == node_id
        neighbors = edge_index[1][mask]
        if len(neighbors) == 0:
            neighbor_embs = embeddings.new_empty((0, embeddings.shape[-1]))
        else:
            neighbor_embs = embeddings[neighbors]
        values.append(float(_embedding_cosine_distance(embeddings[node_id], neighbor_embs)))
    return values


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "q05": _percentile(ordered, 0.05),
        "q25": _percentile(ordered, 0.25),
        "q50": _percentile(ordered, 0.50),
        "q75": _percentile(ordered, 0.75),
        "q95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _degeneracy_flags(
    *,
    distributions: dict[str, dict[str, Any]],
    raw_metric: dict[str, Any],
    supports_detector_signal: bool,
) -> dict[str, bool]:
    signal = distributions["detector_signal"]
    strength = distributions["detector_signal_strength"]
    counter = distributions["counter_signal"]
    quantiles = raw_metric.get("quantiles", {})
    raw_constant = False
    if raw_metric.get("available") and quantiles:
        raw_constant = isclose(
            float(quantiles["min"]),
            float(quantiles["max"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    return {
        "unsupported_detector_signal": not supports_detector_signal,
        "all_detector_signal_unavailable": signal["counts"].get("unavailable", 0)
        == signal["count"]
        and signal["count"] > 0,
        "single_detector_signal": signal["num_unique"] <= 1 and signal["count"] > 0,
        "single_strength": strength["num_unique"] <= 1 and strength["count"] > 0,
        "single_counter_signal": counter["num_unique"] <= 1 and counter["count"] > 0,
        "dominant_detector_signal": signal["dominant_ratio"] >= 0.95
        and signal["count"] > 0,
        "dominant_strength": strength["dominant_ratio"] >= 0.95
        and strength["count"] > 0,
        "raw_metric_unavailable": not raw_metric.get("available", False),
        "raw_metric_constant": raw_constant,
    }


def _score_blind_audit(
    meps: list[MinimalEvidencePackage], raw_metric: dict[str, Any]
) -> dict[str, Any]:
    violations = []
    for mep in meps:
        payload = mep.reasoning.model_dump()
        key_paths = _flatten_keys(payload)
        found = [
            path for path in key_paths
            if path.split(".")[-1].lower() in FORBIDDEN_SCORE_KEYS
        ]
        if found:
            violations.append({"node_id": mep.node_id, "keys": found})
    raw_metric_forbidden_keys = [
        key for key in _flatten_keys(raw_metric)
        if key.split(".")[-1].lower() in FORBIDDEN_SCORE_KEYS
    ]
    return {
        "ok": not violations and not raw_metric_forbidden_keys,
        "num_checked": len(meps),
        "forbidden_score_keys": sorted(FORBIDDEN_SCORE_KEYS),
        "reasoning_violations": violations[:20],
        "raw_metric_forbidden_keys": raw_metric_forbidden_keys,
        "calibration_channel_excluded": True,
    }


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

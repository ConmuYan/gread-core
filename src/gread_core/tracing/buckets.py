"""Bucket assignment for trace node selection.

Three buckets based on prediction score, uncertainty, and labels:
- uncertain: near decision boundary or high uncertainty
- high_conf_fraud: label=1, high score, low uncertainty
- high_conf_benign: label=0, low score, low uncertainty
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor

BucketLabel = Literal["uncertain", "high_conf_fraud", "high_conf_benign"]
BucketPolicy = Literal["fixed", "percentile"]


def _clamp_fraction(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _quantile(values: Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().quantile(_clamp_fraction(q)).item())


def assign_buckets(
    scores: Tensor,
    uncertainties: Tensor,
    labels: Tensor | None,
    bucket_policy: BucketPolicy = "fixed",
    uncertain_threshold: float = 0.15,
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
    uncertainty_high: float = 0.4,
    uncertainty_low: float = 0.2,
    percentile_uncertain_fraction: float = 0.2,
    percentile_high_conf_fraction: float = 0.2,
    percentile_low_uncertainty_fraction: float = 0.5,
) -> list[BucketLabel | None]:
    """Assign each node to a trace bucket based on thresholds.

    Bucket logic:
    fixed:
    - uncertain: abs(score - 0.5) < uncertain_threshold OR uncertainty > uncertainty_high
    - high_conf_fraud: label=1 AND score > high_threshold AND uncertainty < uncertainty_low
    - high_conf_benign: label=0 AND score < low_threshold AND uncertainty < uncertainty_low

    percentile:
    - uncertain: closest-to-boundary scores or highest uncertainties by percentile.
    - high_conf_fraud: label=1, top relative scores, and low relative uncertainty.
    - high_conf_benign: label=0, bottom relative scores, and low relative uncertainty.

    Nodes matching no bucket get None.

    Args:
        scores: prediction scores in [0, 1], shape [N].
        uncertainties: uncertainty values in [0, 1], shape [N].
        labels: optional ground-truth labels (0 or 1), shape [N].
        bucket_policy: "fixed" for absolute thresholds or "percentile" for
            distribution-relative thresholds.
        uncertain_threshold: distance from 0.5 to qualify as uncertain.
        high_threshold: minimum score for high-confidence fraud.
        low_threshold: maximum score for high-confidence benign.
        uncertainty_high: uncertainty above which a node is uncertain.
        uncertainty_low: uncertainty below which high-confidence applies.
        percentile_uncertain_fraction: fraction selected as uncertain by
            boundary proximity or high uncertainty in percentile mode.
        percentile_high_conf_fraction: label-compatible fraction selected for
            fraud/benign score extremity in percentile mode.
        percentile_low_uncertainty_fraction: fraction treated as relatively low
            uncertainty for high-confidence buckets in percentile mode.

    Returns:
        List of bucket labels or None for each node.
    """
    n = scores.shape[0]
    bucket_labels: list[BucketLabel | None] = [None] * n
    if bucket_policy not in ("fixed", "percentile"):
        raise ValueError(f"Unsupported trace bucket policy: {bucket_policy}")

    if bucket_policy == "percentile":
        uncertain_fraction = _clamp_fraction(percentile_uncertain_fraction)
        high_conf_fraction = _clamp_fraction(percentile_high_conf_fraction)
        low_uncertainty_fraction = _clamp_fraction(percentile_low_uncertainty_fraction)
        boundary_distances = (scores.float() - 0.5).abs()
        near_boundary_max = _quantile(boundary_distances, uncertain_fraction)
        high_uncertainty_min = _quantile(uncertainties, 1.0 - uncertain_fraction)
        low_uncertainty_max = _quantile(uncertainties, low_uncertainty_fraction)

        fraud_score_min = 1.0
        benign_score_max = 0.0
        if labels is not None:
            fraud_scores = scores[labels == 1]
            benign_scores = scores[labels == 0]
            fraud_score_min = _quantile(fraud_scores, 1.0 - high_conf_fraction)
            benign_score_max = _quantile(benign_scores, high_conf_fraction)

        for i in range(n):
            s = scores[i].item()
            u = uncertainties[i].item()
            label = labels[i].item() if labels is not None else None

            is_uncertain = uncertain_fraction > 0.0 and (
                abs(s - 0.5) <= near_boundary_max or u >= high_uncertainty_min
            )
            if is_uncertain:
                bucket_labels[i] = "uncertain"
                continue

            if (
                label is not None
                and label == 1
                and s >= fraud_score_min
                and u <= low_uncertainty_max
            ):
                bucket_labels[i] = "high_conf_fraud"
                continue

            if (
                label is not None
                and label == 0
                and s <= benign_score_max
                and u <= low_uncertainty_max
            ):
                bucket_labels[i] = "high_conf_benign"
                continue

        return bucket_labels

    for i in range(n):
        s = scores[i].item()
        u = uncertainties[i].item()
        label = labels[i].item() if labels is not None else None

        # Uncertain: near decision boundary OR high uncertainty
        if abs(s - 0.5) < uncertain_threshold or u > uncertainty_high:
            bucket_labels[i] = "uncertain"
            continue

        # High-confidence fraud: needs label=1, high score, low uncertainty
        if label is not None and label == 1 and s > high_threshold and u < uncertainty_low:
            bucket_labels[i] = "high_conf_fraud"
            continue

        # High-confidence benign: needs label=0, low score, low uncertainty
        if label is not None and label == 0 and s < low_threshold and u < uncertainty_low:
            bucket_labels[i] = "high_conf_benign"
            continue

    return bucket_labels

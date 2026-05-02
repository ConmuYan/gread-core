"""Bucket assignment for trace node selection.

Three buckets based on prediction score, uncertainty, and labels:
- uncertain: near decision boundary or high uncertainty
- high_conf_fraud: label=1, high score, low uncertainty
- high_conf_benign: label=0, low score, low uncertainty
"""

from __future__ import annotations

import logging
from typing import Literal

from torch import Tensor

logger = logging.getLogger(__name__)

BucketLabel = Literal["uncertain", "high_conf_fraud", "high_conf_benign"]


def assign_buckets(
    scores: Tensor,
    uncertainties: Tensor,
    labels: Tensor | None,
    uncertain_threshold: float = 0.15,
    high_threshold: float = 0.7,
    low_threshold: float = 0.3,
    uncertainty_high: float = 0.4,
    uncertainty_low: float = 0.2,
) -> list[BucketLabel | None]:
    """Assign each node to a trace bucket based on thresholds.

    Bucket logic:
    - uncertain: abs(score - 0.5) < uncertain_threshold OR uncertainty > uncertainty_high
    - high_conf_fraud: label=1 AND score > high_threshold AND uncertainty < uncertainty_low
    - high_conf_benign: label=0 AND score < low_threshold AND uncertainty < uncertainty_low

    Nodes matching no bucket get None.

    Args:
        scores: prediction scores in [0, 1], shape [N].
        uncertainties: uncertainty values in [0, 1], shape [N].
        labels: optional ground-truth labels (0 or 1), shape [N].
        uncertain_threshold: distance from 0.5 to qualify as uncertain.
        high_threshold: minimum score for high-confidence fraud.
        low_threshold: maximum score for high-confidence benign.
        uncertainty_high: uncertainty above which a node is uncertain.
        uncertainty_low: uncertainty below which high-confidence applies.

    Returns:
        List of bucket labels or None for each node.
    """
    n = scores.shape[0]
    bucket_labels: list[BucketLabel | None] = [None] * n

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

    # Log empty buckets
    counts: dict[str, int] = {"uncertain": 0, "high_conf_fraud": 0, "high_conf_benign": 0}
    for bl in bucket_labels:
        if bl is not None:
            counts[bl] += 1
    for name, count in counts.items():
        if count == 0:
            logger.warning("Bucket '%s' is empty (0 eligible nodes)", name)

    return bucket_labels

"""Continuous-to-discrete quantization for evidence signals.

Maps continuous values to discrete levels using config-driven thresholds.
All quantization is deterministic given the same thresholds.
"""

from __future__ import annotations

from typing import Literal

# Default thresholds for generic signal quantization.
# Each threshold defines the upper bound for a level.
# Levels are checked in order: first matching threshold wins.

DEFAULT_DEGREE_THRESHOLDS: dict[str, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    # >= 0.75 → high; top percentile → burst (handled separately)
}

DEFAULT_CONSISTENCY_THRESHOLDS: dict[str, float] = {
    "low": 0.33,
    "medium": 0.66,
    "high": 1.01,
}

DEFAULT_DISCREPANCY_THRESHOLDS: dict[str, float] = {
    "low": 0.33,
    "medium": 0.66,
    "high": 1.01,
}

DEFAULT_UNCERTAINTY_THRESHOLDS: dict[str, float] = {
    "low": 0.33,
    "medium": 0.66,
    "high": 1.01,
}

DegreeLevel = Literal["isolated", "low", "medium", "high", "burst"]
ConsistencyLevel = Literal["unavailable", "low", "medium", "high"]
DiscrepancyLevel = Literal["unavailable", "low", "medium", "high"]
UncertaintyLevel = Literal["low", "medium", "high"]


def quantize(value: float, thresholds: dict[str, float]) -> str:
    """Quantize a continuous value to a discrete level.

    Thresholds map level names to upper bounds. Levels are evaluated in
    insertion order; the first level whose threshold exceeds the value wins.
    If no threshold matches (value >= all thresholds), the last level is returned.

    Args:
        value: Continuous value in [0, 1] (or any range matching thresholds).
        thresholds: Mapping from level name to upper-bound threshold.

    Returns:
        The discrete level string.

    Raises:
        ValueError: If thresholds is empty.
    """
    if not thresholds:
        raise ValueError("thresholds must be non-empty")

    for level, upper in thresholds.items():
        if value < upper:
            return level
    # value >= all thresholds → return last level
    return list(thresholds.keys())[-1]


def quantize_degree_level(
    normalized_degrees: list[float],
    thresholds: dict[str, float] | None = None,
    burst_percentile: float = 0.99,
) -> list[str]:
    """Quantize normalized degree values to discrete levels.

    Special handling:
    - 0.0 → "isolated" (no neighbors)
    - >= burst_percentile → "burst"
    - Otherwise → quantized via thresholds

    Args:
        normalized_degrees: Degree values normalized to [0, 1].
        thresholds: Custom thresholds. Defaults to DEFAULT_DEGREE_THRESHOLDS.
        burst_percentile: Percentile above which a node is "burst".

    Returns:
        List of degree level strings.
    """
    if thresholds is None:
        thresholds = DEFAULT_DEGREE_THRESHOLDS

    if not normalized_degrees:
        return []

    # Compute burst threshold from percentile
    sorted_vals = sorted(normalized_degrees)
    burst_idx = int(len(sorted_vals) * burst_percentile)
    burst_idx = min(burst_idx, len(sorted_vals) - 1)
    burst_threshold = sorted_vals[burst_idx]

    levels: list[str] = []
    for deg in normalized_degrees:
        if deg == 0.0:
            levels.append("isolated")
        elif deg >= burst_threshold and burst_threshold > 0:
            levels.append("burst")
        else:
            levels.append(quantize(deg, thresholds))
    return levels


def quantize_consistency(
    values: list[float],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Quantize neighbor consistency values.

    Args:
        values: Consistency values in [0, 1]. NaN → "unavailable".
        thresholds: Custom thresholds. Defaults to DEFAULT_CONSISTENCY_THRESHOLDS.

    Returns:
        List of consistency level strings.
    """
    if thresholds is None:
        thresholds = DEFAULT_CONSISTENCY_THRESHOLDS

    levels: list[str] = []
    for v in values:
        if v != v:  # NaN check
            levels.append("unavailable")
        else:
            levels.append(quantize(v, thresholds))
    return levels


def quantize_discrepancy(
    values: list[float],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Quantize feature-neighbor discrepancy values.

    Args:
        values: Discrepancy values in [0, 1]. NaN → "unavailable".
        thresholds: Custom thresholds. Defaults to DEFAULT_DISCREPANCY_THRESHOLDS.

    Returns:
        List of discrepancy level strings.
    """
    if thresholds is None:
        thresholds = DEFAULT_DISCREPANCY_THRESHOLDS

    levels: list[str] = []
    for v in values:
        if v != v:  # NaN check
            levels.append("unavailable")
        else:
            levels.append(quantize(v, thresholds))
    return levels


def quantize_uncertainty(
    values: list[float],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Quantize uncertainty values.

    Args:
        values: Uncertainty values in [0, 1].
        thresholds: Custom thresholds. Defaults to DEFAULT_UNCERTAINTY_THRESHOLDS.

    Returns:
        List of uncertainty level strings.
    """
    if thresholds is None:
        thresholds = DEFAULT_UNCERTAINTY_THRESHOLDS

    return [quantize(v, thresholds) for v in values]

"""Tests for continuous-to-discrete quantization.

Validates:
- Quantization is deterministic and config-driven
- Thresholds from YAML are respected
- Edge cases handled correctly
"""

from __future__ import annotations

import pytest

from gread_core.evidence.quantization import (
    DEFAULT_CONSISTENCY_THRESHOLDS,
    DEFAULT_DEGREE_THRESHOLDS,
    DEFAULT_DISCREPANCY_THRESHOLDS,
    DEFAULT_UNCERTAINTY_THRESHOLDS,
    quantize,
    quantize_consistency,
    quantize_degree_level,
    quantize_discrepancy,
    quantize_uncertainty,
)

# ---------------------------------------------------------------------------
# quantize (core function)
# ---------------------------------------------------------------------------

class TestQuantize:
    def test_low_value(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66}
        assert quantize(0.1, thresholds) == "low"

    def test_medium_value(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66}
        assert quantize(0.5, thresholds) == "medium"

    def test_above_last_threshold(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66}
        assert quantize(0.8, thresholds) == "medium"

    def test_boundary_low_medium(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66}
        assert quantize(0.33, thresholds) == "medium"

    def test_boundary_medium_high(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66, "high": 1.0}
        assert quantize(0.66, thresholds) == "high"

    def test_exact_threshold_boundary(self) -> None:
        thresholds = {"a": 0.5, "b": 1.0}
        assert quantize(0.499, thresholds) == "a"
        assert quantize(0.5, thresholds) == "b"

    def test_deterministic(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66, "high": 1.0}
        results = [quantize(0.5, thresholds) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_empty_thresholds_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            quantize(0.5, {})

    def test_value_exceeds_all_thresholds(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66, "high": 1.0}
        assert quantize(1.5, thresholds) == "high"

    def test_zero_value(self) -> None:
        thresholds = {"low": 0.33, "medium": 0.66}
        assert quantize(0.0, thresholds) == "low"

    def test_custom_thresholds_from_yaml(self) -> None:
        """Simulates loading thresholds from a YAML config."""
        yaml_thresholds = {
            "weak": 0.25,
            "moderate": 0.50,
            "strong": 0.75,
        }
        assert quantize(0.1, yaml_thresholds) == "weak"
        assert quantize(0.3, yaml_thresholds) == "moderate"
        assert quantize(0.6, yaml_thresholds) == "strong"
        assert quantize(0.8, yaml_thresholds) == "strong"


# ---------------------------------------------------------------------------
# quantize_degree_level
# ---------------------------------------------------------------------------

class TestQuantizeDegreeLevel:
    def test_empty_input(self) -> None:
        assert quantize_degree_level([]) == []

    def test_all_zero_isolated(self) -> None:
        result = quantize_degree_level([0.0, 0.0, 0.0])
        assert all(r == "isolated" for r in result)

    def test_burst_detection(self) -> None:
        values = [0.1, 0.2, 0.3, 0.9, 0.95, 1.0]
        result = quantize_degree_level(values, burst_percentile=0.5)
        # Sorted: [0.1, 0.2, 0.3, 0.9, 0.95, 1.0], 50% → index 3 → threshold 0.9
        assert result[3] == "burst"
        assert result[4] == "burst"
        assert result[5] == "burst"

    def test_mixed_values(self) -> None:
        values = [0.0, 0.1, 0.5, 0.9]
        result = quantize_degree_level(values, burst_percentile=0.99)
        assert result[0] == "isolated"
        assert len(result) == 4

    def test_valid_levels_only(self) -> None:
        values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        result = quantize_degree_level(values)
        valid = {"isolated", "low", "medium", "high", "burst"}
        for level in result:
            assert level in valid, f"Invalid level: {level}"


# ---------------------------------------------------------------------------
# quantize_consistency
# ---------------------------------------------------------------------------

class TestQuantizeConsistency:
    def test_empty_input(self) -> None:
        assert quantize_consistency([]) == []

    def test_nan_becomes_unavailable(self) -> None:
        result = quantize_consistency([float("nan")])
        assert result == ["unavailable"]

    def test_zero_consistency(self) -> None:
        result = quantize_consistency([0.0])
        assert result[0] == "low"

    def test_perfect_consistency(self) -> None:
        result = quantize_consistency([1.0])
        assert result[0] == "high"

    def test_mixed_values(self) -> None:
        values = [float("nan"), 0.0, 0.5, 1.0]
        result = quantize_consistency(values)
        assert result[0] == "unavailable"
        assert result[1] == "low"
        assert result[2] == "medium"
        assert result[3] == "high"

    def test_valid_levels_only(self) -> None:
        values = [float("nan"), 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        result = quantize_consistency(values)
        valid = {"unavailable", "low", "medium", "high"}
        for level in result:
            assert level in valid


# ---------------------------------------------------------------------------
# quantize_discrepancy
# ---------------------------------------------------------------------------

class TestQuantizeDiscrepancy:
    def test_empty_input(self) -> None:
        assert quantize_discrepancy([]) == []

    def test_nan_becomes_unavailable(self) -> None:
        result = quantize_discrepancy([float("nan")])
        assert result == ["unavailable"]

    def test_zero_discrepancy(self) -> None:
        result = quantize_discrepancy([0.0])
        assert result[0] == "low"

    def test_high_discrepancy(self) -> None:
        result = quantize_discrepancy([0.9])
        assert result[0] == "high"

    def test_valid_levels_only(self) -> None:
        values = [float("nan"), 0.0, 0.2, 0.5, 0.8, 1.0]
        result = quantize_discrepancy(values)
        valid = {"unavailable", "low", "medium", "high"}
        for level in result:
            assert level in valid


# ---------------------------------------------------------------------------
# quantize_uncertainty
# ---------------------------------------------------------------------------

class TestQuantizeUncertainty:
    def test_empty_input(self) -> None:
        assert quantize_uncertainty([]) == []

    def test_zero_uncertainty(self) -> None:
        result = quantize_uncertainty([0.0])
        assert result[0] == "low"

    def test_max_uncertainty(self) -> None:
        result = quantize_uncertainty([1.0])
        assert result[0] == "high"

    def test_medium_uncertainty(self) -> None:
        result = quantize_uncertainty([0.5])
        assert result[0] == "medium"

    def test_valid_levels_only(self) -> None:
        values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        result = quantize_uncertainty(values)
        valid = {"low", "medium", "high"}
        for level in result:
            assert level in valid


# ---------------------------------------------------------------------------
# Default thresholds exist and are ordered
# ---------------------------------------------------------------------------

class TestDefaultThresholds:
    def test_all_defaults_defined(self) -> None:
        assert len(DEFAULT_DEGREE_THRESHOLDS) > 0
        assert len(DEFAULT_CONSISTENCY_THRESHOLDS) > 0
        assert len(DEFAULT_DISCREPANCY_THRESHOLDS) > 0
        assert len(DEFAULT_UNCERTAINTY_THRESHOLDS) > 0

    def test_defaults_are_ordered(self) -> None:
        """Threshold values should be monotonically increasing."""
        for name, thresholds in [
            ("degree", DEFAULT_DEGREE_THRESHOLDS),
            ("consistency", DEFAULT_CONSISTENCY_THRESHOLDS),
            ("discrepancy", DEFAULT_DISCREPANCY_THRESHOLDS),
            ("uncertainty", DEFAULT_UNCERTAINTY_THRESHOLDS),
        ]:
            vals = list(thresholds.values())
            for i in range(1, len(vals)):
                assert vals[i] >= vals[i - 1], (
                    f"{name} thresholds not ordered: {vals}"
                )

    def test_custom_thresholds_override(self) -> None:
        custom = {"low": 0.5, "high": 1.0}
        assert quantize(0.3, custom) == "low"
        assert quantize(0.7, custom) == "high"

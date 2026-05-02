"""Unit tests for reasoning quality metrics."""

from __future__ import annotations

import pytest

from gread_core.evaluation.reasoning import (
    compute_acceptance_rate,
    compute_all_reasoning_metrics,
    compute_evidence_f1,
    compute_risk_type_agreement,
)


class TestAcceptanceRate:
    def test_all_accepted(self) -> None:
        assert compute_acceptance_rate(10, 10) == pytest.approx(1.0)

    def test_none_accepted(self) -> None:
        assert compute_acceptance_rate(0, 10) == pytest.approx(0.0)

    def test_half(self) -> None:
        assert compute_acceptance_rate(5, 10) == pytest.approx(0.5)

    def test_zero_total(self) -> None:
        assert compute_acceptance_rate(0, 0) == 0.0


class TestEvidenceF1:
    def test_identical_sets(self) -> None:
        assert compute_evidence_f1(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_disjoint_sets(self) -> None:
        assert compute_evidence_f1(["a", "b"], ["c", "d"]) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        f1 = compute_evidence_f1(["a", "b", "c"], ["b", "c", "d"])
        # intersection = {b, c} -> precision = 2/3, recall = 2/3
        assert f1 == pytest.approx(2 / 3)

    def test_both_empty(self) -> None:
        assert compute_evidence_f1([], []) == pytest.approx(1.0)

    def test_one_empty(self) -> None:
        assert compute_evidence_f1(["a"], []) == pytest.approx(0.0)
        assert compute_evidence_f1([], ["a"]) == pytest.approx(0.0)

    def test_subset(self) -> None:
        # predicted is subset of reference
        f1 = compute_evidence_f1(["a"], ["a", "b", "c"])
        # precision = 1/1 = 1, recall = 1/3
        expected = 2 * 1 * (1 / 3) / (1 + 1 / 3)
        assert f1 == pytest.approx(expected)


class TestRiskTypeAgreement:
    def test_same_type(self) -> None:
        assert compute_risk_type_agreement("camouflage_neighbor", "camouflage_neighbor")

    def test_different_type(self) -> None:
        assert not compute_risk_type_agreement("camouflage_neighbor", "spectral_anomaly")


class TestAllReasoningMetrics:
    def test_perfect_predictions(self) -> None:
        preds = [
            {"accepted": True, "evidence": ["a", "b"], "risk_type": "camouflage_neighbor"},
            {"accepted": True, "evidence": ["c"], "risk_type": "spectral_anomaly"},
        ]
        refs = [
            {"accepted": True, "evidence": ["a", "b"], "risk_type": "camouflage_neighbor"},
            {"accepted": True, "evidence": ["c"], "risk_type": "spectral_anomaly"},
        ]
        metrics = compute_all_reasoning_metrics(preds, refs)
        assert metrics["acceptance_rate"] == pytest.approx(1.0)
        assert metrics["evidence_f1"] == pytest.approx(1.0)
        assert metrics["risk_type_accuracy"] == pytest.approx(1.0)

    def test_empty(self) -> None:
        metrics = compute_all_reasoning_metrics([], [])
        assert metrics["acceptance_rate"] == 0.0
        assert metrics["evidence_f1"] == 0.0
        assert metrics["risk_type_accuracy"] == 0.0

    def test_mixed(self) -> None:
        preds = [
            {"accepted": False, "evidence": ["x"], "risk_type": "wrong"},
        ]
        refs = [
            {"accepted": True, "evidence": ["a"], "risk_type": "correct"},
        ]
        metrics = compute_all_reasoning_metrics(preds, refs)
        assert metrics["acceptance_rate"] == pytest.approx(0.0)
        assert metrics["risk_type_accuracy"] == pytest.approx(0.0)

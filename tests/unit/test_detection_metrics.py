"""Unit tests for detection metrics."""

from __future__ import annotations

import numpy as np
import pytest

from gread_core.evaluation.detection import (
    compute_all_detection_metrics,
    compute_auc,
    compute_auprc,
    compute_f1,
    compute_precision_at_k,
    compute_recall_at_k,
)


class TestAUC:
    def test_perfect_separation(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_score) == pytest.approx(1.0)

    def test_random_performance(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=1000)
        y_score = rng.uniform(0, 1, size=1000)
        auc = compute_auc(y_true, y_score)
        assert 0.4 < auc < 0.6  # near random

    def test_all_positive(self) -> None:
        y_true = np.array([1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7])
        # single class -> sklearn raises or returns specific value
        with pytest.raises(ValueError):
            compute_auc(y_true, y_score)


class TestAUPRC:
    def test_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auprc(y_true, y_score) == pytest.approx(1.0)

    def test_imbalanced(self) -> None:
        y_true = np.array([0, 0, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        auprc = compute_auprc(y_true, y_score)
        assert auprc > 0.5


class TestF1:
    def test_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_f1(y_true, y_pred) == pytest.approx(1.0)

    def test_zero(self) -> None:
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert compute_f1(y_true, y_pred) == pytest.approx(0.0)

    def test_all_wrong(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert compute_f1(y_true, y_pred) == pytest.approx(0.0)


class TestPrecisionAtK:
    def test_top_k_all_positive(self) -> None:
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert compute_precision_at_k(y_true, y_score, k=2) == pytest.approx(1.0)

    def test_top_k_none_positive(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert compute_precision_at_k(y_true, y_score, k=2) == pytest.approx(0.0)

    def test_k_zero(self) -> None:
        y_true = np.array([1, 0])
        y_score = np.array([0.9, 0.1])
        assert compute_precision_at_k(y_true, y_score, k=0) == 0.0

    def test_k_larger_than_n(self) -> None:
        y_true = np.array([1, 0, 1])
        y_score = np.array([0.9, 0.5, 0.8])
        # k=10 but only 3 items: uses all 3
        assert compute_precision_at_k(y_true, y_score, k=10) == pytest.approx(2.0 / 3)


class TestRecallAtK:
    def test_captures_all_positives(self) -> None:
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        assert compute_recall_at_k(y_true, y_score, k=2) == pytest.approx(1.0)

    def test_captures_half(self) -> None:
        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.3, 0.8, 0.2, 0.1])
        assert compute_recall_at_k(y_true, y_score, k=2) == pytest.approx(0.5)

    def test_no_positives(self) -> None:
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7])
        assert compute_recall_at_k(y_true, y_score, k=2) == 0.0

    def test_k_zero(self) -> None:
        y_true = np.array([1, 0])
        y_score = np.array([0.9, 0.1])
        assert compute_recall_at_k(y_true, y_score, k=0) == 0.0


class TestAllDetectionMetrics:
    def test_returns_all_keys(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_all_detection_metrics(y_true, y_score, y_pred)
        assert "auc" in metrics
        assert "auprc" in metrics
        assert "f1" in metrics
        assert "precision@10" in metrics
        assert "recall@10" in metrics

    def test_custom_k_values(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_all_detection_metrics(
            y_true, y_score, y_pred, k_values=[1, 2]
        )
        assert "precision@1" in metrics
        assert "recall@2" in metrics

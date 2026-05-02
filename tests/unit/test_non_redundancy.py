"""Unit tests for non-redundancy evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from gread_core.evaluation.non_redundancy import (
    compute_non_redundancy,
    fit_score_only_model,
    fit_score_type_evidence_model,
    fit_score_type_model,
)


class TestScoreOnlyModel:
    def test_returns_required_keys(self) -> None:
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = fit_score_only_model(scores, labels)
        assert "auc" in result
        assert "auprc" in result
        assert "coef" in result
        assert result["model_name"] == "Y~P"

    def test_perfect_separation(self) -> None:
        scores = np.array([0.0, 0.1, 0.9, 1.0])
        labels = np.array([0, 0, 1, 1])
        result = fit_score_only_model(scores, labels)
        assert result["auc"] == pytest.approx(1.0, abs=0.05)


class TestScoreTypeModel:
    def test_returns_required_keys(self) -> None:
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        types = np.eye(6)[:4]
        labels = np.array([0, 0, 1, 1])
        result = fit_score_type_model(scores, types, labels)
        assert "auc" in result
        assert result["model_name"] == "Y~P+T"


class TestScoreTypeEvidenceModel:
    def test_returns_required_keys(self) -> None:
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        types = np.eye(6)[:4]
        evidence = np.random.RandomState(42).uniform(0, 1, (4, 8))
        labels = np.array([0, 0, 1, 1])
        result = fit_score_type_evidence_model(scores, types, evidence, labels)
        assert "auc" in result
        assert result["model_name"] == "Y~P+T+M"


class TestComputeNonRedundancy:
    def test_returns_all_components(self) -> None:
        rng = np.random.RandomState(42)
        n = 100
        scores = rng.uniform(0, 1, n)
        types = np.eye(6)[rng.randint(0, 6, n)]
        evidence = rng.uniform(0, 1, (n, 8))
        labels = rng.randint(0, 2, n)

        result = compute_non_redundancy(scores, types, evidence, labels)

        assert "models" in result
        assert len(result["models"]) == 3
        assert "auc_improvement" in result
        assert "auprc_improvement" in result
        assert "type_over_score" in result["auc_improvement"]
        assert "evidence_over_type" in result["auc_improvement"]

    def test_model_names_ordered(self) -> None:
        rng = np.random.RandomState(42)
        n = 50
        scores = rng.uniform(0, 1, n)
        types = np.eye(6)[rng.randint(0, 6, n)]
        evidence = rng.uniform(0, 1, (n, 4))
        labels = rng.randint(0, 2, n)

        result = compute_non_redundancy(scores, types, evidence, labels)
        names = [m["model_name"] for m in result["models"]]
        assert names == ["Y~P", "Y~P+T", "Y~P+T+M"]

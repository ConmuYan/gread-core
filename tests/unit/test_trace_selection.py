"""Tests for trace node selection: bucket assignment and diversity sampling."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor

from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)
from gread_core.tracing.buckets import assign_buckets
from gread_core.tracing.diversity import diversity_sample
from gread_core.tracing.selector import SelectionResult, TraceSelector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mep(
    node_id: int, detector_signal: str = "neutral"
) -> MinimalEvidencePackage:
    return MinimalEvidencePackage(
        node_id=str(node_id),
        detector_name="test",
        calibration=CalibrationChannel(
            prediction_score=0.5, uncertainty=0.3
        ),
        reasoning=ReasoningChannel(
            uncertainty_level="medium",
            degree_level="medium",
            neighbor_consistency="medium",
            feature_neighbor_discrepancy="low",
            detector_signal=detector_signal,
            detector_signal_strength="weak",
            counter_signal="test_counter",
            allowed_support_ids=[
                "degree_level",
                "neighbor_consistency",
                "feature_neighbor_discrepancy",
                "detector_signal",
            ],
            allowed_counter_ids=["counter_signal", "uncertainty_level"],
        ),
    )


def _make_scores_uncertainties(
    n: int = 30,
) -> tuple[Tensor, Tensor]:
    torch.manual_seed(42)
    scores = torch.rand(n)
    uncertainties = torch.rand(n)
    return scores, uncertainties


# ---------------------------------------------------------------------------
# Bucket assignment tests
# ---------------------------------------------------------------------------

class TestAssignBuckets:
    def test_returns_list_of_length_n(self) -> None:
        scores, uncertainties = _make_scores_uncertainties(10)
        result = assign_buckets(scores, uncertainties, labels=None)
        assert len(result) == 10

    def test_valid_bucket_labels_or_none(self) -> None:
        scores, uncertainties = _make_scores_uncertainties(20)
        result = assign_buckets(scores, uncertainties, labels=None)
        valid = {"uncertain", "high_conf_fraud", "high_conf_benign", None}
        for label in result:
            assert label in valid

    def test_high_uncertainty_goes_to_uncertain(self) -> None:
        scores = torch.tensor([0.5, 0.9])
        uncertainties = torch.tensor([0.5, 0.5])
        result = assign_buckets(scores, uncertainties, labels=None)
        # Both have high uncertainty (0.5 > 0.4) -> uncertain
        assert result[0] == "uncertain"
        assert result[1] == "uncertain"

    def test_near_boundary_goes_to_uncertain(self) -> None:
        scores = torch.tensor([0.45, 0.55])
        uncertainties = torch.tensor([0.1, 0.1])
        result = assign_buckets(scores, uncertainties, labels=None)
        # |0.45 - 0.5| = 0.05 < 0.15 -> uncertain
        assert result[0] == "uncertain"
        assert result[1] == "uncertain"

    def test_high_score_label_1_low_unc_fraud(self) -> None:
        scores = torch.tensor([0.9])
        uncertainties = torch.tensor([0.1])
        labels = torch.tensor([1])
        result = assign_buckets(scores, uncertainties, labels=labels)
        assert result[0] == "high_conf_fraud"

    def test_low_score_label_0_low_unc_benign(self) -> None:
        scores = torch.tensor([0.1])
        uncertainties = torch.tensor([0.1])
        labels = torch.tensor([0])
        result = assign_buckets(scores, uncertainties, labels=labels)
        assert result[0] == "high_conf_benign"

    def test_no_label_no_fraud_benign(self) -> None:
        scores = torch.tensor([0.9])
        uncertainties = torch.tensor([0.1])
        result = assign_buckets(scores, uncertainties, labels=None)
        # Without labels, can't be high_conf_fraud or high_conf_benign
        assert result[0] is None

    def test_empty_input(self) -> None:
        scores = torch.tensor([])
        uncertainties = torch.tensor([])
        result = assign_buckets(scores, uncertainties, labels=None)
        assert result == []


# ---------------------------------------------------------------------------
# Diversity sampling tests
# ---------------------------------------------------------------------------

class TestDiversitySample:
    def test_respects_budget(self) -> None:
        candidates = list(range(20))
        meps = [_make_mep(i) for i in range(20)]
        rng = random.Random(42)
        selected, scores = diversity_sample(candidates, meps, budget=5, rng=rng)
        assert len(selected) == 5
        assert len(scores) == 5

    def test_returns_all_when_budget_exceeds_candidates(self) -> None:
        candidates = [0, 1, 2]
        meps = [_make_mep(i) for i in range(3)]
        rng = random.Random(42)
        selected, _scores = diversity_sample(candidates, meps, budget=10, rng=rng)
        assert sorted(selected) == [0, 1, 2]

    def test_empty_candidates(self) -> None:
        meps: list[MinimalEvidencePackage] = []
        rng = random.Random(42)
        selected, scores = diversity_sample([], meps, budget=5, rng=rng)
        assert selected == []
        assert scores == []

    def test_zero_budget(self) -> None:
        candidates = [0, 1, 2]
        meps = [_make_mep(i) for i in range(3)]
        rng = random.Random(42)
        selected, _scores = diversity_sample(candidates, meps, budget=0, rng=rng)
        assert selected == []

    def test_deterministic_under_seed(self) -> None:
        candidates = list(range(20))
        meps = [_make_mep(i, f"signal_{i % 3}") for i in range(20)]
        s1, sc1 = diversity_sample(candidates, meps, 5, random.Random(99))
        s2, sc2 = diversity_sample(candidates, meps, 5, random.Random(99))
        assert s1 == s2
        assert sc1 == sc2

    def test_diversity_prefers_unique_signals(self) -> None:
        candidates = list(range(6))
        signals = ["sig_a", "sig_a", "sig_b", "sig_b", "sig_c", "sig_c"]
        meps = [_make_mep(i, signals[i]) for i in range(6)]
        rng = random.Random(42)
        selected, _scores = diversity_sample(candidates, meps, budget=3, rng=rng)
        selected_signals = {signals[i] for i in selected}
        assert len(selected_signals) == 3

    def test_first_node_gets_score_one(self) -> None:
        candidates = [0, 1, 2]
        meps = [_make_mep(i, f"sig_{i}") for i in range(3)]
        rng = random.Random(42)
        _selected, scores = diversity_sample(candidates, meps, budget=3, rng=rng)
        assert scores[0] == 1.0


# ---------------------------------------------------------------------------
# TraceSelector tests
# ---------------------------------------------------------------------------

class TestTraceSelector:
    def _make_config(self, budget: int = 100) -> dict:
        return {
            "trace_selection": {
                "total_budget": budget,
                "buckets": {
                    "uncertain": 0.333,
                    "high_conf_fraud": 0.333,
                    "high_conf_benign": 0.334,
                },
                "diversity_sampling": True,
            }
        }

    def test_returns_selection_result(self) -> None:
        selector = TraceSelector(self._make_config(), seed=42)
        scores, uncertainties = _make_scores_uncertainties(30)
        meps = [_make_mep(i) for i in range(30)]
        result = selector.select(scores, uncertainties, None, meps)
        assert isinstance(result, SelectionResult)

    def test_result_has_required_fields(self) -> None:
        selector = TraceSelector(self._make_config(), seed=42)
        scores, uncertainties = _make_scores_uncertainties(30)
        meps = [_make_mep(i) for i in range(30)]
        result = selector.select(scores, uncertainties, None, meps)
        assert hasattr(result, "node_ids")
        assert hasattr(result, "bucket_labels")
        assert hasattr(result, "diversity_scores")

    def test_node_ids_not_empty(self) -> None:
        selector = TraceSelector(self._make_config(), seed=42)
        scores = torch.cat([torch.full((10,), 0.5)])
        uncertainties = torch.full((10,), 0.6)  # all uncertain
        meps = [_make_mep(i) for i in range(10)]
        result = selector.select(scores, uncertainties, None, meps)
        assert len(result.node_ids) > 0

    def test_total_does_not_exceed_budget(self) -> None:
        budget = 50
        selector = TraceSelector(self._make_config(budget), seed=42)
        scores, uncertainties = _make_scores_uncertainties(200)
        meps = [_make_mep(i) for i in range(200)]
        result = selector.select(scores, uncertainties, None, meps)
        assert len(result.node_ids) <= budget

    def test_deterministic_under_seed(self) -> None:
        scores, uncertainties = _make_scores_uncertainties(30)
        meps = [_make_mep(i) for i in range(30)]
        r1 = TraceSelector(
            self._make_config(), seed=42
        ).select(scores, uncertainties, None, meps)
        r2 = TraceSelector(
            self._make_config(), seed=42
        ).select(scores, uncertainties, None, meps)
        assert r1.node_ids == r2.node_ids

    def test_bucket_labels_match_node_ids(self) -> None:
        selector = TraceSelector(self._make_config(), seed=42)
        scores, uncertainties = _make_scores_uncertainties(30)
        meps = [_make_mep(i) for i in range(30)]
        result = selector.select(scores, uncertainties, None, meps)
        assert len(result.node_ids) == len(result.bucket_labels)
        assert len(result.node_ids) == len(result.diversity_scores)

    def test_no_diversity_sampling(self) -> None:
        config = self._make_config()
        config["trace_selection"]["diversity_sampling"] = False
        selector = TraceSelector(config, seed=42)
        scores = torch.full((20,), 0.5)
        uncertainties = torch.full((20,), 0.6)
        meps = [_make_mep(i) for i in range(20)]
        result = selector.select(scores, uncertainties, None, meps)
        assert len(result.node_ids) > 0

    def test_empty_input_returns_empty_result(self) -> None:
        selector = TraceSelector(self._make_config(), seed=42)
        scores = torch.tensor([])
        uncertainties = torch.tensor([])
        meps: list[MinimalEvidencePackage] = []
        result = selector.select(scores, uncertainties, None, meps)
        assert result.node_ids == []

    def test_empty_buckets_handled_gracefully(self) -> None:
        """All uncertain -> no fraud/benign nodes, but result still valid."""
        selector = TraceSelector(self._make_config(100), seed=42)
        scores = torch.full((20,), 0.5)
        uncertainties = torch.full((20,), 0.9)
        meps = [_make_mep(i) for i in range(20)]
        result = selector.select(scores, uncertainties, None, meps)
        # All should be uncertain
        assert all(bl == "uncertain" for bl in result.bucket_labels)

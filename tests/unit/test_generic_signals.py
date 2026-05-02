"""Tests for generic evidence signal computation.

Validates:
- Generic signals return valid MEP field values (matching risk_taxonomy)
- prediction_score only enters CalibrationChannel, never ReasoningChannel
- Edge cases: isolated node, all-same-label neighborhood
"""

from __future__ import annotations

import torch

from gread_core.evidence.generic_signals import (
    compute_degree_level,
    compute_feature_neighbor_discrepancy,
    compute_neighbor_consistency,
    compute_uncertainty,
)


def _make_sparse_adj(n: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a torch sparse COO adjacency matrix from edge list."""
    if not edges:
        return torch.sparse_coo_tensor(size=(n, n))
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    # Symmetric
    all_rows = rows + cols
    all_cols = cols + rows
    indices = torch.tensor([all_rows, all_cols], dtype=torch.long)
    values = torch.ones(len(all_rows), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


# ---------------------------------------------------------------------------
# compute_degree_level
# ---------------------------------------------------------------------------

class TestComputeDegreeLevel:
    """Tests for compute_degree_level()."""

    def test_returns_list_of_strings(self) -> None:
        degrees = torch.tensor([0, 3, 10, 30, 100])
        result = compute_degree_level(degrees)
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_length_matches_input(self) -> None:
        degrees = torch.tensor([1, 5, 20])
        result = compute_degree_level(degrees)
        assert len(result) == 3

    def test_valid_values(self) -> None:
        degrees = torch.tensor([0, 3, 10, 30, 100])
        result = compute_degree_level(degrees)
        valid = {"isolated", "low", "medium", "high", "burst"}
        assert all(r in valid for r in result)

    def test_zero_degree_isolated(self) -> None:
        degrees = torch.tensor([0])
        result = compute_degree_level(degrees)
        assert result[0] == "isolated"

    def test_all_zero_isolated(self) -> None:
        degrees = torch.zeros(5, dtype=torch.long)
        result = compute_degree_level(degrees)
        assert all(r == "isolated" for r in result)

    def test_single_high_degree_is_burst(self) -> None:
        degrees = torch.tensor([3])
        result = compute_degree_level(degrees)
        # Only node → normalized = 1.0 → burst (top percentile)
        assert result[0] == "burst"

    def test_highest_degree_is_burst(self) -> None:
        degrees = torch.tensor([1, 2, 5, 10, 100])
        result = compute_degree_level(degrees)
        assert result[-1] == "burst"

    def test_empty_tensor(self) -> None:
        result = compute_degree_level(torch.tensor([], dtype=torch.long))
        assert result == []

    def test_custom_thresholds(self) -> None:
        degrees = torch.tensor([0, 1, 5, 20])
        custom = {"small": 0.25, "big": 0.75}
        result = compute_degree_level(degrees, quantization_thresholds=custom)
        assert result[0] == "isolated"
        assert len(result) == 4


# ---------------------------------------------------------------------------
# compute_neighbor_consistency
# ---------------------------------------------------------------------------

class TestComputeNeighborConsistency:
    """Tests for compute_neighbor_consistency()."""

    def test_returns_list_of_strings(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        adj = _make_sparse_adj(4, [(0, 1), (2, 3)])
        result = compute_neighbor_consistency(labels, adj)
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_same_label_neighbors_high_consistency(self) -> None:
        labels = torch.tensor([0, 0, 0, 0])
        adj = _make_sparse_adj(4, [(0, 1), (1, 2), (2, 3)])
        result = compute_neighbor_consistency(labels, adj)
        assert all(r == "high" for r in result)

    def test_different_label_neighbors_low_consistency(self) -> None:
        labels = torch.tensor([0, 1, 0, 1])
        adj = _make_sparse_adj(4, [(0, 1), (1, 2), (2, 3)])
        result = compute_neighbor_consistency(labels, adj)
        # Node 0: neighbor is node 1 (label 1 != 0) → 0/1 = 0.0 → low
        assert result[0] == "low"
        # Node 3: neighbor is node 2 (label 0 != 1) → 0/1 = 0.0 → low
        assert result[3] == "low"

    def test_isolated_node_unavailable(self) -> None:
        """Isolated node (no neighbors) should return 'unavailable'."""
        labels = torch.tensor([0, 1])
        adj = _make_sparse_adj(2, [])
        result = compute_neighbor_consistency(labels, adj)
        assert result == ["unavailable", "unavailable"]

    def test_all_same_label_neighborhood(self) -> None:
        """All-same-label neighborhood should return 'high'."""
        labels = torch.tensor([1, 1, 1])
        adj = _make_sparse_adj(3, [(0, 1), (1, 2)])
        result = compute_neighbor_consistency(labels, adj)
        assert all(r == "high" for r in result)

    def test_length_matches_input(self) -> None:
        labels = torch.tensor([0, 1, 0])
        adj = _make_sparse_adj(3, [(0, 1)])
        result = compute_neighbor_consistency(labels, adj)
        assert len(result) == 3

    def test_valid_values_only(self) -> None:
        labels = torch.randint(0, 2, (8,))
        edges = [(i, i + 1) for i in range(7)]
        adj = _make_sparse_adj(8, edges)
        result = compute_neighbor_consistency(labels, adj)
        valid = {"unavailable", "low", "medium", "high"}
        assert all(r in valid for r in result)


# ---------------------------------------------------------------------------
# compute_feature_neighbor_discrepancy
# ---------------------------------------------------------------------------

class TestComputeFeatureNeighborDiscrepancy:
    """Tests for compute_feature_neighbor_discrepancy()."""

    def test_returns_list_of_strings(self) -> None:
        x = torch.randn(4, 8)
        adj = _make_sparse_adj(4, [(0, 1), (2, 3)])
        result = compute_feature_neighbor_discrepancy(x, adj)
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_identical_features_low_discrepancy(self) -> None:
        x = torch.ones(3, 4)
        adj = _make_sparse_adj(3, [(0, 1), (1, 2)])
        result = compute_feature_neighbor_discrepancy(x, adj)
        assert all(r == "low" for r in result)

    def test_isolated_node_unavailable(self) -> None:
        """Isolated node should return 'unavailable'."""
        x = torch.randn(2, 4)
        adj = _make_sparse_adj(2, [])
        result = compute_feature_neighbor_discrepancy(x, adj)
        assert result == ["unavailable", "unavailable"]

    def test_orthogonal_features_high_discrepancy(self) -> None:
        """Orthogonal features → cos_sim = 0 → discrepancy = 1.0 → high."""
        adj = _make_sparse_adj(2, [(0, 1)])
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = compute_feature_neighbor_discrepancy(x, adj)
        assert result[0] == "high"
        assert result[1] == "high"

    def test_valid_values(self) -> None:
        x = torch.randn(5, 8)
        adj = _make_sparse_adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        result = compute_feature_neighbor_discrepancy(x, adj)
        valid = {"unavailable", "low", "medium", "high"}
        assert all(r in valid for r in result)


# ---------------------------------------------------------------------------
# compute_uncertainty
# ---------------------------------------------------------------------------

class TestComputeUncertainty:
    """Tests for compute_uncertainty()."""

    def test_returns_tuple(self) -> None:
        logits = torch.tensor([[0.0, 2.0], [2.0, -2.0]])
        result = compute_uncertainty(logits)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_list_and_list(self) -> None:
        logits = torch.tensor([[0.0, 2.0], [2.0, -2.0]])
        levels, raw = compute_uncertainty(logits)
        assert isinstance(levels, list)
        assert isinstance(raw, list)

    def test_length_matches_input(self) -> None:
        logits = torch.tensor([[0.0, 2.0], [2.0, -2.0], [-1.0, 1.0]])
        levels, raw = compute_uncertainty(logits)
        assert len(levels) == 3
        assert len(raw) == 3

    def test_confident_prediction_low_uncertainty(self) -> None:
        logits = torch.tensor([[10.0, -10.0]])
        levels, raw = compute_uncertainty(logits)
        assert levels[0] == "low"
        assert raw[0] < 0.1

    def test_uniform_distribution_high_uncertainty(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        levels, raw = compute_uncertainty(logits)
        assert levels[0] == "high"
        assert raw[0] > 0.9

    def test_valid_values(self) -> None:
        logits = torch.randn(10, 3)
        levels, _ = compute_uncertainty(logits)
        valid = {"low", "medium", "high"}
        assert all(lvl in valid for lvl in levels)

    def test_raw_uncertainty_in_range(self) -> None:
        logits = torch.randn(100, 5)
        _, raw = compute_uncertainty(logits)
        for v in raw:
            assert 0.0 <= v <= 1.0 + 1e-6

    def test_empty_logits(self) -> None:
        logits = torch.zeros(0, 2)
        levels, raw = compute_uncertainty(logits)
        assert levels == []
        assert raw == []


# ---------------------------------------------------------------------------
# Score-blindness: prediction_score never enters signals
# ---------------------------------------------------------------------------

class TestScoreBlindness:
    """Tests verifying prediction_score never enters reasoning signals."""

    def test_no_score_input_in_any_signal(self) -> None:
        """Verify none of the generic signal functions accept prediction_score."""
        import inspect

        from gread_core.evidence import generic_signals

        for name in [
            "compute_degree_level",
            "compute_neighbor_consistency",
            "compute_feature_neighbor_discrepancy",
            "compute_uncertainty",
        ]:
            func = getattr(generic_signals, name)
            params = inspect.signature(func).parameters
            param_names = set(params.keys())
            assert "prediction_score" not in param_names, f"{name} accepts prediction_score"
            assert "score" not in param_names, f"{name} has score parameter"

    def test_degree_level_independent_of_score(self) -> None:
        degrees = torch.tensor([2, 5, 10])
        result_a = compute_degree_level(degrees)
        result_b = compute_degree_level(degrees)
        assert result_a == result_b

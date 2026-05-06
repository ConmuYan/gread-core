"""Tests for base GNN detectors (M4 milestone).

Tests:
- Every detector's forward pass produces (logit, embedding) with correct shapes.
- Model has detector_name attribute.
- Embedding is pre-head (not logit).
- Each detector satisfies the DetectorProtocol.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from gread_core.detectors import (
    DetectorProtocol,
    GATDetector,
    GCNDetector,
    GINDetector,
    GPRGNNDetector,
    H2GCNDetector,
    PCGNNDetector,
    SAGEDetector,
)


def _make_graph(num_nodes: int = 10, num_features: int = 4) -> Any:
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7, 0]],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index)


def test_gcn_detector_forward() -> None:
    model = GCNDetector(in_channels=4, hidden_channels=8, num_layers=2)
    graph = _make_graph(num_nodes=10, num_features=4)
    logit, embedding = model.forward_with_embedding(graph)

    assert logit.shape == (10,)
    assert embedding.shape == (10, 8)
    assert model.detector_name == "gcn"


def test_gcn_detector_protocol_compliance() -> None:
    model = GCNDetector(in_channels=4, hidden_channels=8)
    assert isinstance(model, DetectorProtocol)


# --------------------------------------------------------------------------- #
# Parametrised smoke tests for the newly added detectors.                     #
# --------------------------------------------------------------------------- #

_HIDDEN = 8
_N = 10
_F = 4


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected_name", "expected_emb_dim"),
    [
        (SAGEDetector, {"num_layers": 2}, "sage", _HIDDEN),
        (GATDetector, {"num_layers": 2, "heads": 2}, "gat", _HIDDEN),
        (GINDetector, {"num_layers": 2}, "gin", _HIDDEN),
        (GPRGNNDetector, {"num_hops": 3}, "gpr_gnn", _HIDDEN),
        # H2GCN emb dim = hidden * (2^{K+1} - 1); K=2 -> 7 * hidden.
        (H2GCNDetector, {"num_rounds": 2}, "h2gcn", _HIDDEN * 7),
        (PCGNNDetector, {"num_layers": 2}, "pc_gnn", _HIDDEN),
    ],
)
def test_detector_forward_shapes(
    cls: type, kwargs: dict, expected_name: str, expected_emb_dim: int
) -> None:
    model = cls(in_channels=_F, hidden_channels=_HIDDEN, **kwargs)
    graph = _make_graph(num_nodes=_N, num_features=_F)
    logit, embedding = model.forward_with_embedding(graph)

    assert logit.shape == (_N,), f"{expected_name} logit shape"
    assert embedding.shape == (_N, expected_emb_dim), f"{expected_name} emb shape"
    assert model.detector_name == expected_name
    assert isinstance(model, DetectorProtocol)


def test_gpr_gnn_exposes_learnable_gammas() -> None:
    """GPR-GNN must expose `self.gammas` as a learnable parameter of size K+1.

    This is the multi-hop evidence signal consumed by the adapter.
    """
    model = GPRGNNDetector(in_channels=_F, hidden_channels=_HIDDEN, num_hops=5)
    assert hasattr(model, "gammas")
    assert isinstance(model.gammas, torch.nn.Parameter)
    assert model.gammas.shape == (6,)
    assert model.gammas.requires_grad


def test_pc_gnn_exposes_layer_scores() -> None:
    """PC-GNN must populate `self.layer_scores` after forward pass.

    Each entry is a per-node CHOOSE-gate score in [0, 1], forming the
    signed-evidence channel for the PC-GNN adapter.
    """
    num_layers = 3
    model = PCGNNDetector(
        in_channels=_F, hidden_channels=_HIDDEN, num_layers=num_layers
    )
    graph = _make_graph(num_nodes=_N, num_features=_F)
    _ = model.forward_with_embedding(graph)

    assert len(model.layer_scores) == num_layers
    for s in model.layer_scores:
        assert s.shape == (_N,)
        assert torch.all(s >= 0.0) and torch.all(s <= 1.0)


def test_h2gcn_handles_edgeless_graph() -> None:
    """H2GCN must degrade gracefully when the graph has no edges.

    With no edges, both 1-hop and 2-hop aggregations are zero, so the
    output is driven purely by the ego embedding.
    """
    x = torch.randn(_N, _F)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index)

    model = H2GCNDetector(in_channels=_F, hidden_channels=_HIDDEN, num_rounds=2)
    logit, embedding = model.forward_with_embedding(graph)

    assert logit.shape == (_N,)
    assert embedding.shape == (_N, _HIDDEN * 7)
    assert torch.isfinite(logit).all()

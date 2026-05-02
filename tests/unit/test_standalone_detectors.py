"""Tests for standalone detector models (BWGNN, CARE-GNN, TreeNeighbor).

Validates that each detector:
- Instantiates without error
- Produces correct output shapes from forward_with_embedding
- Follows the DetectorProtocol contract
"""

import torch
from torch_geometric.data import Data

from gread_core.detectors.bwgnn import BWGNNDetector
from gread_core.detectors.caregnn import CAREGNNDetector
from gread_core.detectors.tree_neighbor import TreeNeighborDetector


def _make_tiny_graph(num_nodes: int = 20, in_channels: int = 16) -> Data:
    """Create a tiny random graph for testing."""
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:10] = True
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[10:] = True
    return Data(x=x, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)


class TestBWGNNDetector:
    def test_init(self) -> None:
        det = BWGNNDetector(in_channels=16, hidden_channels=32)
        assert det.detector_name == "bwgnn"

    def test_forward_shape(self) -> None:
        det = BWGNNDetector(in_channels=16, hidden_channels=32)
        graph = _make_tiny_graph(in_channels=16)
        logit, emb = det.forward_with_embedding(graph)
        assert logit.shape == (10,)  # test_mask selects 10 nodes
        assert emb.shape == (20, 32)  # full graph embeddings

    def test_no_mask_returns_all(self) -> None:
        det = BWGNNDetector(in_channels=16, hidden_channels=32)
        graph = Data(x=torch.randn(20, 16), edge_index=torch.randint(0, 20, (2, 40)))
        logit, emb = det.forward_with_embedding(graph)
        assert logit.shape == (20,)
        assert emb.shape == (20, 32)


class TestCAREGNNDetector:
    def test_init(self) -> None:
        det = CAREGNNDetector(in_channels=16, hidden_channels=32)
        assert det.detector_name == "caregnn"

    def test_forward_shape(self) -> None:
        det = CAREGNNDetector(in_channels=16, hidden_channels=32)
        graph = _make_tiny_graph(in_channels=16)
        logit, emb = det.forward_with_embedding(graph)
        assert logit.shape == (10,)
        assert emb.shape == (20, 32)

    def test_no_mask_returns_all(self) -> None:
        det = CAREGNNDetector(in_channels=16, hidden_channels=32)
        graph = Data(x=torch.randn(20, 16), edge_index=torch.randint(0, 20, (2, 40)))
        logit, emb = det.forward_with_embedding(graph)
        assert logit.shape == (20,)
        assert emb.shape == (20, 32)


class TestTreeNeighborDetector:
    def test_init(self) -> None:
        det = TreeNeighborDetector(in_channels=16, hidden_channels=32)
        assert det.detector_name == "tree_neighbor"

    def test_forward_shape(self) -> None:
        det = TreeNeighborDetector(in_channels=16, hidden_channels=32)
        graph = _make_tiny_graph(in_channels=16)
        logit, emb = det.forward_with_embedding(graph)
        assert logit.shape == (10,)
        assert emb.shape == (20, 64)  # hidden*2 (self + neighbor concat)

    def test_no_mask_returns_all(self) -> None:
        det = TreeNeighborDetector(in_channels=16, hidden_channels=32)
        graph = Data(x=torch.randn(20, 16), edge_index=torch.randint(0, 20, (2, 40)))
        logit, emb = det.forward_with_embedding(graph)
        assert logit.shape == (20,)
        assert emb.shape == (20, 64)  # hidden*2 (self + neighbor concat)

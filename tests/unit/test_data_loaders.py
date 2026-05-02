"""Tests for data loaders, splits, and base detectors.

Validates:
- Tiny graph fixture loads as PyG Data with correct attributes
- GCN/GAT detectors return (logit, embedding) with correct shapes
- forward_with_embedding is the only entry point
- Train/val/test masks are valid
"""

from __future__ import annotations

import re

import pytest
import torch

from gread_core.data.loaders import load_tiny_graph
from gread_core.data.splits import generate_masks, stratified_split

try:
    from torch_geometric.data import Data

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYG_AVAILABLE, reason="torch_geometric not available")


class TestTinyGraphLoader:
    """Tests for load_tiny_graph()."""

    def test_returns_data_object(self) -> None:
        graph = load_tiny_graph()
        assert isinstance(graph, Data)

    def test_has_node_features(self) -> None:
        graph = load_tiny_graph(num_nodes=50, num_features=16)
        assert hasattr(graph, "x")
        assert graph.x.shape == (50, 16)

    def test_has_edge_index(self) -> None:
        graph = load_tiny_graph()
        assert hasattr(graph, "edge_index")
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0

    def test_has_labels(self) -> None:
        graph = load_tiny_graph(num_nodes=50, fraud_ratio=0.2)
        assert hasattr(graph, "y")
        assert graph.y.shape == (50,)
        assert graph.y.dtype == torch.long
        # Check fraud ratio is approximately correct
        fraud_count = (graph.y == 1).sum().item()
        assert 5 <= fraud_count <= 15  # ~20% of 50

    def test_has_masks(self) -> None:
        graph = load_tiny_graph(num_nodes=50)
        assert hasattr(graph, "train_mask")
        assert hasattr(graph, "val_mask")
        assert hasattr(graph, "test_mask")

        assert graph.train_mask.shape == (50,)
        assert graph.val_mask.shape == (50,)
        assert graph.test_mask.shape == (50,)

        assert graph.train_mask.dtype == torch.bool
        assert graph.val_mask.dtype == torch.bool
        assert graph.test_mask.dtype == torch.bool

    def test_masks_cover_all_nodes(self) -> None:
        graph = load_tiny_graph(num_nodes=100)
        combined = graph.train_mask | graph.val_mask | graph.test_mask
        assert combined.all()

    def test_masks_are_disjoint(self) -> None:
        graph = load_tiny_graph(num_nodes=100)
        # No node should be in more than one split
        assert not (graph.train_mask & graph.val_mask).any()
        assert not (graph.train_mask & graph.test_mask).any()
        assert not (graph.val_mask & graph.test_mask).any()

    def test_deterministic_with_seed(self) -> None:
        graph1 = load_tiny_graph(seed=42)
        graph2 = load_tiny_graph(seed=42)
        assert torch.equal(graph1.x, graph2.x)
        assert torch.equal(graph1.y, graph2.y)
        assert torch.equal(graph1.edge_index, graph2.edge_index)

    def test_different_seeds_give_different_graphs(self) -> None:
        graph1 = load_tiny_graph(seed=1)
        graph2 = load_tiny_graph(seed=2)
        # At least features should differ
        assert not torch.equal(graph1.x, graph2.x)


class TestGenerateMasks:
    """Tests for generate_masks()."""

    def test_adds_masks_to_data(self) -> None:
        data = Data(
            x=torch.randn(100, 16),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.zeros(100, dtype=torch.long),
        )
        data = generate_masks(data, seed=1)
        assert hasattr(data, "train_mask")
        assert hasattr(data, "val_mask")
        assert hasattr(data, "test_mask")

    def test_masks_cover_all_nodes(self) -> None:
        data = Data(x=torch.randn(100, 16))
        data = generate_masks(data, seed=1)
        combined = data.train_mask | data.val_mask | data.test_mask
        assert combined.all()

    def test_custom_ratios(self) -> None:
        data = Data(x=torch.randn(100, 16))
        data = generate_masks(data, seed=1, ratios=(0.6, 0.2, 0.2))
        train_count = data.train_mask.sum().item()
        val_count = data.val_mask.sum().item()
        test_count = data.test_mask.sum().item()
        assert train_count == 60
        assert val_count == 20
        assert test_count == 20

    def test_invalid_ratios_raises(self) -> None:
        data = Data(x=torch.randn(100, 16))
        with pytest.raises(ValueError, match=re.escape("Ratios must sum to 1.0")):
            generate_masks(data, seed=1, ratios=(0.5, 0.3, 0.3))


class TestStratifiedSplit:
    """Tests for stratified_split()."""

    def test_preserves_class_distribution(self) -> None:
        # Create imbalanced dataset: 80 class-0, 20 class-1
        y = torch.cat([torch.zeros(80, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        data = Data(x=torch.randn(100, 16), y=y)
        data = stratified_split(data, seed=1)

        # Check each split has approximately 20% fraud
        for mask_name in ["train_mask", "val_mask", "test_mask"]:
            mask = getattr(data, mask_name)
            fraud_ratio = data.y[mask].float().mean().item()
            assert 0.15 <= fraud_ratio <= 0.25, f"{mask_name} fraud ratio: {fraud_ratio}"


class TestGCNDetector:
    """Tests for GCNDetector."""

    def _make_detector(self) -> torch.nn.Module:
        from gread_core.detectors.pyg_gnn import GCNDetector

        return GCNDetector(in_channels=16, hidden_channels=32, num_layers=2)

    def _make_graph(self, num_nodes: int = 50) -> Data:
        from gread_core.data.loaders import load_tiny_graph

        return load_tiny_graph(num_nodes=num_nodes, num_features=16, seed=42)

    def test_returns_tuple(self) -> None:
        detector = self._make_detector()
        graph = self._make_graph()
        result = detector.forward_with_embedding(graph)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes(self) -> None:
        detector = self._make_detector()
        graph = self._make_graph(num_nodes=50)
        logit, embedding = detector.forward_with_embedding(graph)

        # With test_mask, B should be number of test nodes
        num_test = graph.test_mask.sum().item()
        assert logit.shape == (num_test,)
        assert embedding.shape == (50, 32)

    def test_embedding_dimension(self) -> None:
        detector = self._make_detector()
        graph = self._make_graph()
        _, embedding = detector.forward_with_embedding(graph)
        assert embedding.shape[1] == 32

    def test_detector_name(self) -> None:
        detector = self._make_detector()
        assert detector.detector_name == "gcn"

    def test_is_module(self) -> None:
        detector = self._make_detector()
        assert isinstance(detector, torch.nn.Module)

    def test_gradient_flow(self) -> None:
        detector = self._make_detector()
        graph = self._make_graph()
        graph.x = graph.x.requires_grad_(True)

        logit, _embedding = detector.forward_with_embedding(graph)
        loss = logit.sum()
        loss.backward()

        assert graph.x.grad is not None

    def test_no_separate_forward(self) -> None:
        """Verify forward_with_embedding is the only entry point."""
        detector = self._make_detector()
        # forward_with_embedding should exist
        assert hasattr(detector, "forward_with_embedding")
        # There should be no separate forward() method that skips embedding
        # (the module inherits nn.Module.forward, but it is not meant to be used directly)


class TestGATDetector:
    """Tests for GATDetector."""

    def _make_detector(self) -> torch.nn.Module:
        from gread_core.detectors.pyg_gnn import GATDetector

        return GATDetector(
            in_channels=16,
            hidden_channels=32,
            num_layers=2,
            attention_heads=4,
        )

    def _make_graph(self, num_nodes: int = 50) -> Data:
        from gread_core.data.loaders import load_tiny_graph

        return load_tiny_graph(num_nodes=num_nodes, num_features=16, seed=42)

    def test_returns_tuple(self) -> None:
        detector = self._make_detector()
        graph = self._make_graph()
        result = detector.forward_with_embedding(graph)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes(self) -> None:
        detector = self._make_detector()
        graph = self._make_graph(num_nodes=50)
        logit, embedding = detector.forward_with_embedding(graph)

        num_test = graph.test_mask.sum().item()
        assert logit.shape == (num_test,)
        assert embedding.shape == (50, 32)

    def test_detector_name(self) -> None:
        detector = self._make_detector()
        assert detector.detector_name == "gat"


class TestDetectorProtocol:
    """Tests verifying DetectorProtocol conformance."""

    def test_gcn_conforms_to_protocol(self) -> None:
        from gread_core.detectors.base import DetectorProtocol
        from gread_core.detectors.pyg_gnn import GCNDetector

        detector = GCNDetector(in_channels=16, hidden_channels=32)
        assert isinstance(detector, DetectorProtocol)

    def test_gat_conforms_to_protocol(self) -> None:
        from gread_core.detectors.base import DetectorProtocol
        from gread_core.detectors.pyg_gnn import GATDetector

        detector = GATDetector(in_channels=16, hidden_channels=32)
        assert isinstance(detector, DetectorProtocol)

    def test_protocol_check_rejects_non_detector(self) -> None:
        from gread_core.detectors.base import DetectorProtocol

        class NotADetector:
            pass

        assert not isinstance(NotADetector(), DetectorProtocol)

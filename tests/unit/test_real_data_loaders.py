"""Tests for real dataset loaders (YelpChi, Amazon, tfinance, tsocial)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from gread_core.data.loaders import (
    load_graph_dataset,
    load_real_amazon,
    load_real_tfinance,
    load_real_tsocial,
    load_real_yelpchi,
)

# Real dataset root
DATA_ROOT = "/data1/mq/codes/awesome-graph-anomaly-detection/PriorF-GNN/datasets"


class TestLoadRealYelpChi:
    """Test YelpChi .mat loader."""

    def test_returns_data_object(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "y")

    def test_node_count(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[0] == 45954

    def test_feature_dim(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[1] == 32

    def test_labels_binary(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        unique_labels = torch.unique(data.y)
        assert all(lbl in [0, 1] for lbl in unique_labels.tolist())

    def test_fraud_ratio_reasonable(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        fraud_ratio = data.y.sum().item() / data.y.shape[0]
        assert 0.01 < fraud_ratio < 0.50

    def test_masks_generated(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        assert data.train_mask is not None
        assert data.val_mask is not None
        assert data.test_mask is not None

    def test_masks_cover_all_nodes(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        total = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
        assert total.item() == data.x.shape[0]

    def test_edge_index_shape(self) -> None:
        data = load_real_yelpchi(data_root=DATA_ROOT, seed=1)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0


class TestLoadRealAmazon:
    """Test Amazon .mat loader."""

    def test_returns_data_object(self) -> None:
        data = load_real_amazon(data_root=DATA_ROOT, seed=1)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "y")

    def test_node_count(self) -> None:
        data = load_real_amazon(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[0] == 11944

    def test_feature_dim(self) -> None:
        data = load_real_amazon(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[1] == 25

    def test_labels_binary(self) -> None:
        data = load_real_amazon(data_root=DATA_ROOT, seed=1)
        unique_labels = torch.unique(data.y)
        assert all(lbl in [0, 1] for lbl in unique_labels.tolist())

    def test_fraud_ratio_reasonable(self) -> None:
        data = load_real_amazon(data_root=DATA_ROOT, seed=1)
        fraud_ratio = data.y.sum().item() / data.y.shape[0]
        assert 0.01 < fraud_ratio < 0.50

    def test_masks_generated(self) -> None:
        data = load_real_amazon(data_root=DATA_ROOT, seed=1)
        assert data.train_mask is not None
        assert data.val_mask is not None
        assert data.test_mask is not None


class TestLoadGraphDataset:
    """Test load_graph_dataset with real data integration."""

    def test_yelpchi_returns_real_data(self) -> None:
        data = load_graph_dataset("yelpchi", seed=1)
        # Real YelpChi has 45954 nodes, synthetic fallback has 4000
        assert data.x.shape[0] == 45954

    def test_amazon_returns_real_data(self) -> None:
        data = load_graph_dataset("amazon", seed=1)
        # Real Amazon has 11944 nodes, synthetic fallback has 5000
        assert data.x.shape[0] == 11944

    def test_tiny_still_works(self) -> None:
        data = load_graph_dataset("tiny", seed=1)
        assert data.x.shape[0] == 50

    def test_synthetic_still_works(self) -> None:
        data = load_graph_dataset("synthetic_small", seed=1)
        assert data.x.shape[0] == 500


class TestLoadRealTfinance:
    """Test tfinance DGL binary loader."""

    def test_returns_data_object(self) -> None:
        data = load_real_tfinance(data_root=DATA_ROOT, seed=1)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "y")

    def test_node_count(self) -> None:
        data = load_real_tfinance(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[0] == 39357

    def test_feature_dim(self) -> None:
        data = load_real_tfinance(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[1] == 10

    def test_labels_binary(self) -> None:
        data = load_real_tfinance(data_root=DATA_ROOT, seed=1)
        unique_labels = torch.unique(data.y)
        assert all(lbl in [0, 1] for lbl in unique_labels.tolist())

    def test_fraud_ratio_reasonable(self) -> None:
        data = load_real_tfinance(data_root=DATA_ROOT, seed=1)
        fraud_ratio = data.y.sum().item() / data.y.shape[0]
        assert 0.01 < fraud_ratio < 0.50

    def test_masks_generated(self) -> None:
        data = load_real_tfinance(data_root=DATA_ROOT, seed=1)
        assert data.train_mask is not None
        assert data.val_mask is not None
        assert data.test_mask is not None


class TestLoadRealTsocial:
    """Test tsocial DGL binary loader."""

    def test_returns_data_object(self) -> None:
        data = load_real_tsocial(data_root=DATA_ROOT, seed=1)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "y")

    def test_node_count(self) -> None:
        data = load_real_tsocial(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[0] == 5781065

    def test_feature_dim(self) -> None:
        data = load_real_tsocial(data_root=DATA_ROOT, seed=1)
        assert data.x.shape[1] == 10

    def test_labels_binary(self) -> None:
        data = load_real_tsocial(data_root=DATA_ROOT, seed=1)
        unique_labels = torch.unique(data.y)
        assert all(lbl in [0, 1] for lbl in unique_labels.tolist())

    def test_fraud_ratio_reasonable(self) -> None:
        data = load_real_tsocial(data_root=DATA_ROOT, seed=1)
        fraud_ratio = data.y.sum().item() / data.y.shape[0]
        assert 0.01 < fraud_ratio < 0.50

    def test_masks_generated(self) -> None:
        data = load_real_tsocial(data_root=DATA_ROOT, seed=1)
        assert data.train_mask is not None
        assert data.val_mask is not None
        assert data.test_mask is not None

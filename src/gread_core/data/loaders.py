"""Graph dataset loaders for GReaD-Core.

All loaders return PyG Data objects with:
- x: node feature tensor [N, F]
- edge_index: edge index tensor [2, E]
- y: node label tensor [N] (0=benign, 1=fraud)
- train_mask, val_mask, test_mask: boolean masks [N]
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any


def load_tiny_graph(
    num_nodes: int = 50,
    num_features: int = 16,
    fraud_ratio: float = 0.2,
    edge_probability: float = 0.3,
    seed: int = 42,
) -> Data:
    """Create a synthetic tiny graph for testing and smoke runs.

    Returns a PyG Data object with random features, edges, and labels.
    The graph is deterministic given the seed.
    """
    generator = torch.Generator().manual_seed(seed)

    x = torch.randn(num_nodes, num_features, generator=generator)

    # Generate labels: ~fraud_ratio fraction labeled as fraud (1)
    num_fraud = max(1, int(num_nodes * fraud_ratio))
    y = torch.zeros(num_nodes, dtype=torch.long)
    fraud_indices = torch.randperm(num_nodes, generator=generator)[:num_fraud]
    y[fraud_indices] = 1

    # Generate random edges (undirected)
    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1, generator=generator).item() < edge_probability:
                src_list.extend([i, j])
                dst_list.extend([j, i])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Generate train/val/test masks (70/15/15 split)
    perm = torch.randperm(num_nodes, generator=generator)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_synthetic_graph(
    num_nodes: int = 1000,
    num_features: int = 64,
    fraud_ratio: float = 0.1,
    avg_degree: int = 10,
    seed: int = 42,
) -> Data:
    """Create a larger synthetic graph for pipeline validation.

    Generates a graph with configurable size and fraud ratio, useful for
    validating the full pipeline when real datasets are unavailable.

    Returns a PyG Data object with features, edges, labels, and masks.
    """
    generator = torch.Generator().manual_seed(seed)

    x = torch.randn(num_nodes, num_features, generator=generator)

    # Generate labels with fraud_ratio
    num_fraud = max(1, int(num_nodes * fraud_ratio))
    y = torch.zeros(num_nodes, dtype=torch.long)
    fraud_indices = torch.randperm(num_nodes, generator=generator)[:num_fraud]
    y[fraud_indices] = 1

    # Generate edges: each node connects to ~avg_degree neighbors
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]
    # Make undirected
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    # Generate train/val/test masks (70/15/15 split)
    perm = torch.randperm(num_nodes, generator=generator)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def load_graph_dataset(
    name: str,
    root: str = "data",
    seed: int = 1,
) -> Data:
    """Load a graph fraud detection dataset by name.

    Supported datasets:
    - yelpchi: YelpChi restaurant review fraud (requires FraudDataset)
    - amazon: Amazon review fraud (requires FraudDataset)
    - tfinance: T-Finance transaction fraud (requires FraudDataset)
    - tsocial: T-Social social network fraud (requires FraudDataset)
    - tiny: synthetic tiny graph (50 nodes) for testing
    - synthetic_small: synthetic graph (500 nodes) for validation
    - synthetic_medium: synthetic graph (2000 nodes) for validation
    - synthetic_large: synthetic graph (5000 nodes) for validation

    Returns a PyG Data object with train/val/test masks.
    """
    if name == "tiny":
        return load_tiny_graph(seed=seed)

    # Synthetic graphs for pipeline validation
    synthetic_sizes = {
        "synthetic_small": (500, 64, 0.1, 10),
        "synthetic_medium": (2000, 64, 0.08, 15),
        "synthetic_large": (5000, 128, 0.05, 20),
    }
    if name in synthetic_sizes:
        n, f, ratio, deg = synthetic_sizes[name]
        return load_synthetic_graph(
            num_nodes=n, num_features=f, fraud_ratio=ratio, avg_degree=deg, seed=seed
        )

    # Try FraudDataset for real fraud datasets
    if name in ("yelpchi", "amazon", "tfinance", "tsocial"):
        try:
            from torch_geometric.datasets import FraudDataset

            dataset = FraudDataset(root=root, name=name)
            data = dataset[0]
            if not hasattr(data, "train_mask") or data.train_mask is None:
                data = _generate_masks(data, seed=seed)
            return data
        except (ImportError, Exception) as e:
            # Fall back to synthetic graph with dataset-appropriate sizes
            fallback_sizes = {
                "yelpchi": (4000, 32, 0.05, 15),
                "amazon": (5000, 25, 0.05, 12),
                "tfinance": (3000, 10, 0.08, 10),
                "tsocial": (6000, 10, 0.05, 15),
            }
            n, f, ratio, deg = fallback_sizes[name]
            import logging
            logging.getLogger(__name__).warning(
                "FraudDataset not available for '%s' (%s). "
                "Using synthetic graph (%d nodes, %d features) as fallback.",
                name, e, n, f,
            )
            return load_synthetic_graph(
                num_nodes=n, num_features=f, fraud_ratio=ratio, avg_degree=deg, seed=seed
            )

    msg = (
        f"Unknown dataset: '{name}'. "
        "Available: tiny, synthetic_small, synthetic_medium, synthetic_large, "
        "yelpchi, amazon, tfinance, tsocial"
    )
    raise ValueError(msg)


def _generate_masks(
    data: Data,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Data:
    """Generate train/val/test masks for a graph dataset.

    Thin wrapper around gread_core.data.splits.generate_masks.
    """
    from gread_core.data.splits import generate_masks as gen_masks

    return gen_masks(data, seed=seed, ratios=ratios)

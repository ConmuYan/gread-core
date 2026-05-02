"""Train/val/test split utilities for graph datasets.

Provides deterministic, seed-based mask generation for PyG Data objects.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any


def generate_masks(
    data: Data,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Data:
    """Generate train/val/test boolean masks for a PyG Data object.

    Creates deterministic masks based on the seed. Masks are added directly
    to the Data object and it is returned.

    Args:
        data: PyG Data object. Must have data.num_nodes or data.x.
        seed: Random seed for reproducibility.
        ratios: (train_ratio, val_ratio, test_ratio). Must sum to 1.0.

    Returns:
        The same Data object with train_mask, val_mask, test_mask added.

    Raises:
        ValueError: If ratios do not sum to 1.0 or data has no nodes.
    """
    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        msg = f"Ratios must sum to 1.0, got {total}"
        raise ValueError(msg)

    num_nodes = _get_num_nodes(data)
    if num_nodes == 0:
        msg = "Data object has no nodes"
        raise ValueError(msg)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def stratified_split(
    data: Data,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    label_col: str = "y",
) -> Data:
    """Generate stratified train/val/test masks preserving class distribution.

    Ensures each split has approximately the same class ratio as the full dataset.

    Args:
        data: PyG Data object with labels.
        seed: Random seed.
        ratios: (train, val, test) ratios.
        label_col: Attribute name for labels.

    Returns:
        Data object with stratified masks added.
    """
    train_ratio, val_ratio, _test_ratio = ratios
    labels = getattr(data, label_col)
    num_nodes = labels.shape[0]

    generator = torch.Generator().manual_seed(seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    unique_labels = labels.unique()
    for label in unique_labels:
        indices = (labels == label).nonzero(as_tuple=True)[0]
        perm = indices[torch.randperm(indices.shape[0], generator=generator)]

        n = perm.shape[0]
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        train_mask[perm[:train_end]] = True
        val_mask[perm[train_end:val_end]] = True
        test_mask[perm[val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def _get_num_nodes(data: Data) -> int:
    """Extract number of nodes from a PyG Data object."""
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        return data.num_nodes  # type: ignore[no-any-return]
    if hasattr(data, "x") and data.x is not None:
        return data.x.shape[0]  # type: ignore[no-any-return]
    if hasattr(data, "y") and data.y is not None:
        return data.y.shape[0]  # type: ignore[no-any-return]
    if hasattr(data, "edge_index") and data.edge_index is not None:
        return int(data.edge_index.max()) + 1
    return 0

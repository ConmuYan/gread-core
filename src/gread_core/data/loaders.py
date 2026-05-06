"""Graph dataset loaders for GReaD-Core.

All loaders return PyG Data objects with:
- x: node feature tensor [N, F]
- edge_index: edge index tensor [2, E]
- y: node label tensor [N] (0=benign, 1=fraud)
- train_mask, val_mask, test_mask: boolean masks [N]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

_DEFAULT_DATA_ROOT = Path("data/raw")
_DATA_ROOT_ENV = "GREAD_DATA_ROOT"


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


def resolve_data_root(
    cli_data_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Resolve the real dataset root.

    Priority: explicit CLI/root argument > config ``data.root``/``data.data_root``
    > ``GREAD_DATA_ROOT`` > project-local ``data/raw`` symlink.
    """
    if cli_data_root:
        return Path(cli_data_root).expanduser()

    data_cfg = config.get("data", {}) if config is not None else {}
    if isinstance(data_cfg, dict):
        configured_root = data_cfg.get("root") or data_cfg.get("data_root")
        if configured_root:
            return Path(str(configured_root)).expanduser()

    env_root = os.environ.get(_DATA_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser()

    return _DEFAULT_DATA_ROOT


def _resolve_split_options(
    config: dict[str, Any] | None = None,
) -> tuple[tuple[float, float, float], bool]:
    split_cfg = (config or {}).get("data", {}).get("split", {})
    ratios_raw = split_cfg.get("ratios", [0.7, 0.15, 0.15])
    ratios = tuple(float(v) for v in ratios_raw)
    if len(ratios) != 3:
        raise ValueError("data.split.ratios must contain exactly 3 values")
    return (ratios[0], ratios[1], ratios[2]), bool(split_cfg.get("stratified", False))


def load_graph_dataset(
    name: str,
    root: str | Path | None = None,
    seed: int = 1,
    allow_synthetic_fallback: bool = False,
    config: dict[str, Any] | None = None,
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

    Args:
        name: Dataset name.
        root: Root directory for real dataset storage. Overrides config/env.
        seed: Random seed for mask generation.
        allow_synthetic_fallback: If True, fall back to synthetic data when
            real dataset is unavailable. If False (default), raise an error
            for real dataset names that cannot be loaded.
        config: Optional config dict with ``data.root`` or ``data.data_root``.

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

    # Try real datasets first, then FraudDataset, then synthetic fallback
    if name in ("yelpchi", "amazon", "tfinance", "tsocial"):
        import logging
        logger = logging.getLogger(__name__)
        data_root = resolve_data_root(cli_data_root=root, config=config)

        # Try loading real dataset from configured/local dataset root
        real_loaders: dict[str, Any] = {
            "yelpchi": load_real_yelpchi,
            "amazon": load_real_amazon,
            "tfinance": load_real_tfinance,
            "tsocial": load_real_tsocial,
        }
        try:
            ratios, stratified = _resolve_split_options(config)
            try:
                return real_loaders[name](
                    data_root=data_root,
                    seed=seed,
                    ratios=ratios,
                    stratified=stratified,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                return real_loaders[name](data_root=data_root, seed=seed)
        except (FileNotFoundError, ImportError, Exception) as e:
            logger.warning("Real dataset '%s' not available: %s", name, e)

        # Try FraudDataset from PyG
        try:
            from torch_geometric.datasets import FraudDataset

            dataset = FraudDataset(root=str(data_root), name=name)
            data = dataset[0]
            if not hasattr(data, "train_mask") or data.train_mask is None:
                ratios, stratified = _resolve_split_options(config)
                data = _generate_masks(
                    data,
                    seed=seed,
                    ratios=ratios,
                    stratified=stratified,
                )
            return data
        except (ImportError, Exception) as e:
            logger.warning("FraudDataset not available for '%s': %s", name, e)

        # Fail closed unless explicitly allowed
        if not allow_synthetic_fallback:
            raise FileNotFoundError(
                f"Real dataset '{name}' could not be loaded. "
                f"Ensure the dataset is available or pass "
                f"allow_synthetic_fallback=True for synthetic approximation."
            )

        # Fall back to synthetic graph with dataset-appropriate sizes
        fallback_sizes = {
            "yelpchi": (4000, 32, 0.05, 15),
            "amazon": (5000, 25, 0.05, 12),
            "tfinance": (3000, 10, 0.08, 10),
            "tsocial": (6000, 10, 0.05, 15),
        }
        n, f, ratio, deg = fallback_sizes[name]
        logger.warning(
            "Using synthetic graph (%d nodes, %d features) as fallback for '%s'.",
            n, f, name,
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
    stratified: bool = False,
) -> Data:
    """Generate train/val/test masks for a graph dataset.

    Thin wrapper around gread_core.data.splits.
    """
    from gread_core.data.splits import generate_masks as gen_masks
    from gread_core.data.splits import stratified_split

    if stratified and hasattr(data, "y") and data.y is not None:
        return stratified_split(data, seed=seed, ratios=ratios)
    return gen_masks(data, seed=seed, ratios=ratios)


def load_real_yelpchi(
    data_root: str | Path | None = None,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratified: bool = False,
) -> Data:
    """Load real YelpChi dataset from .mat file.

    Args:
        data_root: Directory containing YelpChi.mat.
        seed: Random seed for train/val/test split.

    Returns:
        PyG Data object with features, edge_index, labels, and masks.
    """
    import logging

    from scipy.io import loadmat

    logger = logging.getLogger(__name__)
    resolved_root = resolve_data_root(data_root)
    mat_path = resolved_root / "YelpChi.mat"
    if not mat_path.exists():
        msg = f"YelpChi.mat not found at {mat_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading real YelpChi dataset from %s", mat_path)
    mat = loadmat(mat_path)

    # Extract features (sparse -> dense)
    features = mat["features"]
    if hasattr(features, "toarray"):
        features = features.toarray()
    x = torch.tensor(features, dtype=torch.float32)

    # Extract labels
    label = mat["label"].squeeze()
    y = torch.tensor(label, dtype=torch.long)

    # Extract adjacency matrix (use 'homo' for homogeneous graph)
    adj = mat["homo"]
    if hasattr(adj, "tocoo"):
        adj = adj.tocoo()
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)

    n_nodes = x.shape[0]
    logger.info(
        "YelpChi: nodes=%d, features=%d, edges=%d, fraud_ratio=%.2f%%",
        n_nodes, x.shape[1], edge_index.shape[1],
        100.0 * y.sum().item() / n_nodes,
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    return _generate_masks(data, seed=seed, ratios=ratios, stratified=stratified)


def load_real_amazon(
    data_root: str | Path | None = None,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratified: bool = False,
) -> Data:
    """Load real Amazon dataset from .mat file.

    Args:
        data_root: Directory containing Amazon.mat.
        seed: Random seed for train/val/test split.

    Returns:
        PyG Data object with features, edge_index, labels, and masks.
    """
    import logging

    from scipy.io import loadmat

    logger = logging.getLogger(__name__)
    resolved_root = resolve_data_root(data_root)
    mat_path = resolved_root / "Amazon.mat"
    if not mat_path.exists():
        msg = f"Amazon.mat not found at {mat_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading real Amazon dataset from %s", mat_path)
    mat = loadmat(mat_path)

    # Extract features (sparse -> dense)
    features = mat["features"]
    if hasattr(features, "toarray"):
        features = features.toarray()
    x = torch.tensor(features, dtype=torch.float32)

    # Extract labels
    label = mat["label"].squeeze()
    y = torch.tensor(label, dtype=torch.long)

    # Extract adjacency matrix (use 'homo' for homogeneous graph)
    adj = mat["homo"]
    if hasattr(adj, "tocoo"):
        adj = adj.tocoo()
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)

    n_nodes = x.shape[0]
    logger.info(
        "Amazon: nodes=%d, features=%d, edges=%d, fraud_ratio=%.2f%%",
        n_nodes, x.shape[1], edge_index.shape[1],
        100.0 * y.sum().item() / n_nodes,
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    return _generate_masks(data, seed=seed, ratios=ratios, stratified=stratified)


def _ensure_dgl_importable() -> None:
    """Stub out dgl.graphbolt when the C++ library is missing.

    DGL 2.1.x bundles graphbolt that expects a .so matching the installed
    PyTorch version.  When there is a mismatch the entire ``import dgl``
    fails.  We work around this by pre-populating ``sys.modules`` with stub
    sub-modules so DGL's own ``__init__`` never tries to load the missing
    library.
    """
    import sys
    import types

    stubs = [
        "dgl.graphbolt",
        "dgl.graphbolt.base",
        "dgl.graphbolt.dataloader",
        "dgl.graphbolt.impl",
        "dgl.graphbolt.impl.legacy_dataset",
        "dgl.graphbolt.impl.ondisk_dataset",
        "dgl.graphbolt.impl.ondisk_metadata",
    ]
    for name in stubs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod


def load_real_tfinance(
    data_root: str | Path | None = None,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratified: bool = False,
) -> Data:
    """Load real tfinance dataset from DGL binary format.

    Args:
        data_root: Directory containing tfinance DGL binary.
        seed: Random seed for train/val/test split.

    Returns:
        PyG Data object with features, edge_index, labels, and masks.
    """
    import logging
    logger = logging.getLogger(__name__)
    resolved_root = resolve_data_root(data_root)
    dgl_path = resolved_root / "tfinance"
    if not dgl_path.exists():
        msg = f"tfinance not found at {dgl_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading real tfinance dataset from %s", dgl_path)

    _ensure_dgl_importable()
    try:
        from dgl.data.utils import load_graphs
    except ImportError as exc:
        msg = "dgl not installed. Run: pip install dgl"
        raise ImportError(msg) from exc

    graphs, _label_dict = load_graphs(str(dgl_path))
    g = graphs[0]

    # Extract features
    features = g.ndata["feature"].float()
    # Apply log(x+1) transform to raw count features
    n_raw_feat = features.shape[1]
    n_log_cols = n_raw_feat
    for col_idx in range(n_raw_feat):
        col = features[:, col_idx]
        if col.min() >= 0.0 and col.max() <= 1.0:
            n_log_cols = col_idx
            break
    if n_log_cols > 0:
        features_log = torch.log1p(features[:, :n_log_cols])
        features = torch.cat([features_log, features[:, n_log_cols:]], dim=1)

    x = features

    # Extract labels (one-hot -> class index)
    label_raw = g.ndata["label"]
    if label_raw.ndim == 2 and label_raw.shape[1] == 2:
        y = label_raw[:, 1].long()
    else:
        y = label_raw.long().squeeze()

    # Extract edges
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0)

    n_nodes = x.shape[0]
    logger.info(
        "tfinance: nodes=%d, features=%d, edges=%d, fraud_ratio=%.2f%%",
        n_nodes, x.shape[1], edge_index.shape[1],
        100.0 * y.sum().item() / n_nodes,
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    return _generate_masks(data, seed=seed, ratios=ratios, stratified=stratified)


def load_real_tsocial(
    data_root: str | Path | None = None,
    seed: int = 1,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    stratified: bool = False,
) -> Data:
    """Load real tsocial dataset from DGL binary format.

    Args:
        data_root: Directory containing tsocial DGL binary.
        seed: Random seed for train/val/test split.

    Returns:
        PyG Data object with features, edge_index, labels, and masks.
    """
    import logging
    logger = logging.getLogger(__name__)
    resolved_root = resolve_data_root(data_root)
    dgl_path = resolved_root / "tsocial"
    if not dgl_path.exists():
        msg = f"tsocial not found at {dgl_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading real tsocial dataset from %s", dgl_path)

    _ensure_dgl_importable()
    try:
        from dgl.data.utils import load_graphs
    except ImportError as exc:
        msg = "dgl not installed. Run: pip install dgl"
        raise ImportError(msg) from exc

    graphs, _label_dict = load_graphs(str(dgl_path))
    g = graphs[0]

    # Extract features
    features = g.ndata["feature"].float()
    # Apply log(x+1) transform to raw count features
    n_raw_feat = features.shape[1]
    n_log_cols = n_raw_feat
    for col_idx in range(n_raw_feat):
        col = features[:, col_idx]
        if col.min() >= 0.0 and col.max() <= 1.0:
            n_log_cols = col_idx
            break
    if n_log_cols > 0:
        features_log = torch.log1p(features[:, :n_log_cols])
        features = torch.cat([features_log, features[:, n_log_cols:]], dim=1)

    x = features

    # Extract labels (one-hot -> class index)
    label_raw = g.ndata["label"]
    if label_raw.ndim == 2 and label_raw.shape[1] == 2:
        y = label_raw[:, 1].long()
    else:
        y = label_raw.long().squeeze()

    # Extract edges
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0)

    n_nodes = x.shape[0]
    logger.info(
        "tsocial: nodes=%d, features=%d, edges=%d, fraud_ratio=%.2f%%",
        n_nodes, x.shape[1], edge_index.shape[1],
        100.0 * y.sum().item() / n_nodes,
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    return _generate_masks(data, seed=seed, ratios=ratios, stratified=stratified)

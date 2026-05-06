from __future__ import annotations

from typing import Any

import torch

ALL_DETECTOR_TYPES: tuple[str, ...] = (
    "gcn",
    "gat",
    "bwgnn",
    "caregnn",
    "tree_neighbor",
    "sage",
    "pc_gnn",
    "h2gcn",
    "gin",
    "gpr_gnn",
)


def _detector_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    detector_cfg = config.get("detector")
    if isinstance(detector_cfg, dict):
        return detector_cfg
    return config


def create_detector(
    detector_type: str,
    *,
    in_channels: int,
    hidden_channels: int = 64,
    config: dict[str, Any] | None = None,
) -> torch.nn.Module:
    cfg = _detector_config(config)
    normalized = detector_type.lower()
    dropout = float(cfg.get("dropout", 0.5))

    if normalized == "gcn":
        from gread_core.detectors.pyg_gnn import GCNDetector
        return GCNDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=dropout,
        )
    if normalized == "gat":
        from gread_core.detectors.pyg_gnn import GATDetector
        return GATDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=int(cfg.get("num_layers", 2)),
            attention_heads=int(cfg.get("attention_heads", cfg.get("heads", 4))),
            dropout=dropout,
        )
    if normalized == "bwgnn":
        from gread_core.detectors.bwgnn import BWGNNDetector
        return BWGNNDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=dropout,
            num_coeffs=int(cfg.get("num_coeffs", 3)),
        )
    if normalized == "caregnn":
        from gread_core.detectors.caregnn import CAREGNNDetector
        return CAREGNNDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_relations=int(cfg.get("num_relations", 2)),
            sim_threshold=float(cfg.get("sim_threshold", 0.5)),
            dropout=dropout,
        )
    if normalized == "tree_neighbor":
        from gread_core.detectors.tree_neighbor import TreeNeighborDetector
        return TreeNeighborDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
    if normalized == "sage":
        from gread_core.detectors.pyg_gnn import SAGEDetector
        return SAGEDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=dropout,
            aggr=str(cfg.get("aggr", "mean")),
        )
    if normalized == "pc_gnn":
        from gread_core.detectors.pc_gnn import PCGNNDetector
        return PCGNNDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=dropout,
            aggr=str(cfg.get("aggr", "mean")),
        )
    if normalized == "h2gcn":
        from gread_core.detectors.h2gcn import H2GCNDetector
        return H2GCNDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_rounds=int(cfg.get("num_rounds", 2)),
            dropout=dropout,
        )
    if normalized == "gin":
        from gread_core.detectors.pyg_gnn import GINDetector
        return GINDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=dropout,
        )
    if normalized == "gpr_gnn":
        from gread_core.detectors.gpr_gnn import GPRGNNDetector
        return GPRGNNDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_hops=int(cfg.get("num_hops", 10)),
            dropout=dropout,
            alpha=float(cfg.get("alpha", 0.1)),
        )
    raise ValueError(f"Unknown detector: {detector_type}")


def get_detector_embedding_dim(
    detector_type: str,
    *,
    hidden_channels: int = 64,
    config: dict[str, Any] | None = None,
) -> int:
    cfg = _detector_config(config)
    normalized = detector_type.lower()
    if normalized == "tree_neighbor":
        return hidden_channels * 2
    if normalized == "h2gcn":
        num_rounds = int(cfg.get("num_rounds", 2))
        return hidden_channels * ((1 << (num_rounds + 1)) - 1)
    return hidden_channels

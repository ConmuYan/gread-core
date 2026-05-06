"""Factory for detector-specific evidence adapters."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from gread_core.adapters.base import EvidenceAdapter
from gread_core.adapters.bwgnn_adapter import BWGNNAdapter
from gread_core.adapters.caregnn_adapter import CAREGNNAdapter
from gread_core.adapters.pyg_gnn_adapter import PyGGNNAdapter
from gread_core.adapters.tree_adapter import TreeAdapter

_EMBEDDING_GNN_DETECTORS = {"gcn", "gat", "sage"}


def create_evidence_adapter(
    *,
    detector_type: str,
    detector: Any,
    graph: Any,
    logits: Tensor,
    embeddings: Tensor | None,
    thresholds: dict[str, float] | None = None,
    strict_detector_signal: bool = True,
) -> EvidenceAdapter:
    """Create an evidence adapter for a detector family.

    Formal experiments should run with ``strict_detector_signal=True`` so a
    detector that cannot expose detector-native evidence fails closed instead
    of silently producing generic-only placeholders.
    """
    normalized = detector_type.lower()
    if normalized in _EMBEDDING_GNN_DETECTORS:
        adapter: EvidenceAdapter = PyGGNNAdapter(
            detector,
            graph,
            logits,
            embeddings,
            thresholds=thresholds,
            detector_name=normalized,
        )
    elif normalized == "bwgnn":
        adapter = BWGNNAdapter(
            detector,
            graph,
            logits,
            spectral_responses=embeddings,
            thresholds=thresholds,
            detector_name=normalized,
        )
    elif normalized == "caregnn":
        adapter = CAREGNNAdapter(
            detector,
            graph,
            logits,
            filter_weights=getattr(detector, "filter_weights", None),
            thresholds=thresholds,
            detector_name=normalized,
        )
    elif normalized == "tree_neighbor":
        adapter = TreeAdapter(
            detector,
            graph,
            logits,
            feature_importance=getattr(detector, "feature_importance", None),
            thresholds=thresholds,
            detector_name=normalized,
        )
    elif normalized == "pc_gnn":
        native_values = _stack_layer_values(getattr(detector, "layer_scores", None))
        adapter = PyGGNNAdapter(
            detector,
            graph,
            logits,
            embeddings,
            thresholds=thresholds,
            native_values=native_values,
            signal_family=normalized,
            detector_name=normalized,
        )
    elif normalized == "gpr_gnn":
        native_values = _broadcast_vector(getattr(detector, "gammas", None), logits.shape[0])
        adapter = PyGGNNAdapter(
            detector,
            graph,
            logits,
            embeddings,
            thresholds=thresholds,
            native_values=native_values,
            signal_family=normalized,
            detector_name=normalized,
        )
    elif normalized == "gin":
        native_values = _stack_layer_values(getattr(detector, "layer_deltas", None))
        adapter = PyGGNNAdapter(
            detector,
            graph,
            logits,
            embeddings,
            thresholds=thresholds,
            native_values=native_values,
            signal_family=normalized,
            detector_name=normalized,
        )
    elif normalized == "h2gcn":
        adapter = PyGGNNAdapter(
            detector,
            graph,
            logits,
            embeddings,
            thresholds=thresholds,
            native_values=_h2gcn_native_values(embeddings),
            signal_family=normalized,
            detector_name=normalized,
        )
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

    if strict_detector_signal and not adapter.supports_detector_signal():
        raise RuntimeError(
            f"Detector '{detector_type}' does not expose detector-native evidence; "
            "formal experiments fail closed instead of using generic-only placeholders."
        )
    return adapter


def _stack_layer_values(values: Any) -> Tensor | None:
    if not values:
        return None
    tensors = [value.detach().float().flatten() for value in values]
    if not tensors:
        return None
    return torch.stack(tensors, dim=-1)


def _broadcast_vector(values: Any, num_nodes: int) -> Tensor | None:
    if values is None:
        return None
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return None
    return cast(Tensor, flat.unsqueeze(0).expand(num_nodes, -1))


def _h2gcn_native_values(embeddings: Tensor | None) -> Tensor | None:
    if embeddings is None or embeddings.numel() == 0:
        return None
    centered = embeddings.detach().float()
    return (centered - centered.mean(dim=0, keepdim=True)).abs()

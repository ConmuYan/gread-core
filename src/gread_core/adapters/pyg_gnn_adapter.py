"""PyG GNN adapter (GCN/GAT): extracts embedding-neighbor discrepancy signals.

Signal mapping:
- detector_signal values: embedding_neighbor_discrepancy_high,
  attention_concentration_high, message_disagreement_high
- Strength derivation: embedding cosine distance
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gread_core.adapters.base import EvidenceAdapter
from gread_core.evidence.generic_signals import (
    compute_degree_level,
    compute_feature_neighbor_discrepancy,
    compute_neighbor_consistency,
    compute_uncertainty,
)
from gread_core.evidence.mep_builder import build_mep
from gread_core.schemas.evidence import EvidenceStrength, MinimalEvidencePackage


def _build_adj(edge_index: Tensor, n: int) -> Tensor:
    """Build dense adjacency matrix from edge_index."""
    adj = torch.zeros(n, n, dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def _embedding_cosine_distance(
    node_emb: Tensor, neighbor_embs: Tensor
) -> float:
    """Compute mean cosine distance between node and neighbor embeddings."""
    if neighbor_embs.numel() == 0:
        return 0.0
    node_norm = node_emb.norm().item()
    if node_norm == 0.0:
        return 0.0
    neighbor_norms = neighbor_embs.norm(dim=-1)
    dot_products = neighbor_embs @ node_emb
    denom = node_norm * neighbor_norms
    valid = denom > 0
    if not valid.any():
        return 0.0
    cos_sim = dot_products[valid] / denom[valid]
    cos_dist = (1.0 - cos_sim).clamp(0.0, 2.0) / 2.0
    return cos_dist.mean().item()  # type: ignore[no-any-return]


def _derive_signal(cos_dist: float) -> tuple[str, EvidenceStrength]:
    """Map cosine distance to detector signal and strength."""
    if cos_dist >= 0.4:
        return "embedding_neighbor_discrepancy_high", "strong"
    if cos_dist >= 0.25:
        return "message_disagreement_high", "moderate"
    if cos_dist >= 0.10:
        return "attention_concentration_high", "weak"
    return "neutral", "weak"


def _build_counter_signal(cos_dist: float) -> str:
    """Build counter-evidence string from embedding analysis."""
    if cos_dist < 0.1:
        return "embedding_neighbor_alignment"
    if cos_dist < 0.25:
        return "moderate_embedding_alignment"
    return "low_embedding_alignment"


class PyGGNNAdapter(EvidenceAdapter):
    """Adapter for GCN/GAT detectors with embedding-based signals."""

    detector_name: str = "pyg_gnn"

    def __init__(
        self,
        detector: Any,
        graph: Any,
        logits: Tensor,
        embeddings: Tensor | None = None,
    ) -> None:
        self._detector = detector
        self._graph = graph
        self._logits = logits
        self._embeddings = embeddings

        edge_index = graph.edge_index
        x = graph.x
        labels = graph.y
        n = x.shape[0]
        adj = _build_adj(edge_index, n)

        degrees = torch.zeros(n, dtype=torch.long)
        ones = torch.ones(edge_index.shape[1], dtype=torch.long)
        degrees.scatter_add_(0, edge_index[0], ones)

        self._degree_levels = compute_degree_level(degrees)
        self._neighbor_consistency = compute_neighbor_consistency(labels, adj)
        self._feature_discrepancy = compute_feature_neighbor_discrepancy(x, adj)
        self._uncertainty_levels, self._raw_uncertainty = compute_uncertainty(logits)

    def _get_neighbors(self, node_id: int) -> list[int]:
        mask = self._graph.edge_index[0] == node_id
        return self._graph.edge_index[1][mask].tolist()  # type: ignore[no-any-return]

    def supports_detector_signal(self) -> bool:
        emb = self._embeddings
        return emb is not None and emb.numel() > 0

    def extract(self, node_ids: list[int]) -> list[MinimalEvidencePackage]:
        meps: list[MinimalEvidencePackage] = []
        for nid in node_ids:
            meps.append(self._extract_single(nid))
        return meps

    def _extract_single(self, node_id: int) -> MinimalEvidencePackage:
        dl = self._degree_levels
        nc = self._neighbor_consistency
        fd = self._feature_discrepancy
        ul = self._uncertainty_levels

        degree_level = dl[node_id] if node_id < len(dl) else "unavailable"
        neighbor_cons = nc[node_id] if node_id < len(nc) else "unavailable"
        feat_disc = fd[node_id] if node_id < len(fd) else "unavailable"
        unc_level = ul[node_id] if node_id < len(ul) else "low"

        detector_signal: str
        strength: EvidenceStrength
        counter_signal: str

        emb = self._embeddings
        has_emb = (
            self.supports_detector_signal()
            and emb is not None
            and node_id < emb.shape[0]
        )
        if has_emb:
            neighbors = self._get_neighbors(node_id)
            if neighbors:
                neighbor_embs = emb[neighbors]  # type: ignore[index]
            else:
                h = emb.shape[-1]  # type: ignore[union-attr]
                neighbor_embs = torch.empty(0, h)
            cos_dist = _embedding_cosine_distance(
                emb[node_id], neighbor_embs  # type: ignore[index]
            )
            detector_signal, strength = _derive_signal(cos_dist)
            counter_signal = _build_counter_signal(cos_dist)
        else:
            detector_signal = "unavailable"
            strength = "unavailable"
            counter_signal = "no_embedding_data"

        if node_id < self._logits.shape[0]:
            logit_i = self._logits[node_id]
            if logit_i.dim() == 0:
                # Binary: scalar logit → sigmoid
                pred_score = torch.sigmoid(logit_i).item()
            else:
                probs = torch.softmax(logit_i.unsqueeze(0), dim=-1)
                pred_score = (
                    probs[0, -1].item()
                    if probs.shape[-1] > 1
                    else probs[0, 0].item()
                )
            raw = self._raw_uncertainty
            uncertainty = raw[node_id] if node_id < len(raw) else 0.0
        else:
            pred_score = 0.0
            uncertainty = 0.0

        return build_mep(
            node_id=str(node_id),
            detector_name=self.detector_name,
            prediction_score=pred_score,
            uncertainty=uncertainty,
            uncertainty_level=(
                unc_level if unc_level in ("low", "medium", "high") else "low"
            ),
            degree_level=degree_level,
            neighbor_consistency=neighbor_cons,
            feature_neighbor_discrepancy=feat_disc,
            detector_signal=detector_signal,
            detector_signal_strength=strength,
            counter_signal=counter_signal,
        )

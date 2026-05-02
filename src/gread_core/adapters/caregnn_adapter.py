"""CARE-GNN adapter: extracts camouflage resistance signals.

Signal mapping:
- detector_signal values: camouflage_neighbor_filter_high,
  neighbor_selection_disagreement_high, relation_aware_camouflage_signal
- Strength derivation: filter disagreement
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


def _filter_disagreement(filter_weights: Tensor) -> float:
    """Compute filter disagreement from CARE-GNN filter weights."""
    if filter_weights.numel() == 0:
        return 0.0
    return filter_weights.var().item()


def _derive_signal(disagreement: float) -> tuple[str, EvidenceStrength]:
    """Map filter disagreement to detector signal and strength."""
    if disagreement >= 0.25:
        return "camouflage_neighbor_filter_high", "strong"
    if disagreement >= 0.10:
        return "neighbor_selection_disagreement_high", "moderate"
    if disagreement >= 0.02:
        return "relation_aware_camouflage_signal", "weak"
    return "neutral", "weak"


def _build_counter_signal(disagreement: float) -> str:
    """Build counter-evidence string from filter analysis."""
    if disagreement < 0.05:
        return "consistent_neighbor_selection"
    if disagreement < 0.15:
        return "moderate_neighbor_agreement"
    return "high_neighbor_agreement"


class CAREGNNAdapter(EvidenceAdapter):
    """Adapter for CARE-GNN detector with camouflage resistance signals."""

    detector_name: str = "caregnn"

    def __init__(
        self,
        detector: Any,
        graph: Any,
        logits: Tensor,
        filter_weights: dict[int, Tensor] | None = None,
    ) -> None:
        self._detector = detector
        self._graph = graph
        self._logits = logits
        self._filter_weights = filter_weights

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

    def supports_detector_signal(self) -> bool:
        return self._filter_weights is not None

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

        fw = self._filter_weights
        if self.supports_detector_signal() and fw is not None and node_id in fw:
            disagreement = _filter_disagreement(fw[node_id])
            detector_signal, strength = _derive_signal(disagreement)
            counter_signal = _build_counter_signal(disagreement)
        else:
            detector_signal = "unavailable"
            strength = "unavailable"
            counter_signal = "no_filter_data"

        if node_id < self._logits.shape[0]:
            probs = torch.softmax(
                self._logits[node_id].unsqueeze(0), dim=-1
            )
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

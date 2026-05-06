"""Tree adapter: extracts feature importance signals from tree-based detectors.

Signal mapping:
- detector_signal values: feature_importance_risk_high,
  neighborhood_aggregation_discrepancy_high
- Strength derivation: feature importance variance
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
    """Build sparse adjacency matrix from edge_index."""
    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
    return torch.sparse_coo_tensor(edge_index, values, (n, n))


def _derive_signal(
    variance: float, thresholds: dict[str, float] | None = None,
) -> tuple[str, EvidenceStrength]:
    """Map feature importance variance to detector signal and strength."""
    t = thresholds or {}
    if variance >= t.get("signal_strong", 0.05):
        return "feature_importance_risk_high", "strong"
    if variance >= t.get("signal_moderate", 0.02):
        return "neighborhood_aggregation_discrepancy_high", "moderate"
    return "neutral", "weak"


def _build_counter_signal(
    variance: float, thresholds: dict[str, float] | None = None,
) -> str:
    """Build counter-evidence string from feature importance analysis."""
    t = thresholds or {}
    if variance < t.get("counter_uniform", 0.01):
        return "uniform_feature_importance"
    if variance < t.get("counter_moderate", 0.03):
        return "moderate_feature_importance_spread"
    return "concentrated_feature_importance"


class TreeAdapter(EvidenceAdapter):
    """Adapter for tree-based detectors (Random Forest, XGBoost, etc.)."""

    detector_name: str = "tree"

    def __init__(
        self,
        detector: Any,
        graph: Any,
        logits: Tensor,
        feature_importance: Tensor | None = None,
        thresholds: dict[str, float] | None = None,
        detector_name: str | None = None,
    ) -> None:
        if detector_name is not None:
            self.detector_name = detector_name
        self._detector = detector
        self._graph = graph
        self._logits = logits
        self._feature_importance = feature_importance
        self._thresholds = thresholds or {}

        edge_index = graph.edge_index
        x = graph.x
        labels = graph.y
        n = x.shape[0]
        adj = _build_adj(edge_index, n)

        degrees = torch.zeros(n, dtype=torch.long, device=edge_index.device)
        ones = torch.ones(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        degrees.scatter_add_(0, edge_index[0], ones)

        self._degree_levels = compute_degree_level(degrees)
        self._neighbor_consistency = compute_neighbor_consistency(labels, adj)
        self._feature_discrepancy = compute_feature_neighbor_discrepancy(x, adj)
        self._uncertainty_levels, self._raw_uncertainty = compute_uncertainty(logits)

    def supports_detector_signal(self) -> bool:
        fi = self._feature_importance
        return fi is not None and fi.numel() > 0

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

        if self.supports_detector_signal():
            assert self._feature_importance is not None
            variance = self._feature_importance.var().item()
            detector_signal, strength = _derive_signal(variance, self._thresholds)
            counter_signal = _build_counter_signal(variance, self._thresholds)
        else:
            detector_signal = "unavailable"
            strength = "unavailable"
            counter_signal = "no_feature_importance_data"

        if node_id < self._logits.shape[0]:
            logit_i = self._logits[node_id]
            if logit_i.dim() == 0:
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

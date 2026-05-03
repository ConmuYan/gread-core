"""BWGNN adapter: extracts high-frequency response signals from spectral analysis.

Signal mapping:
- detector_signal values: high_frequency_response_high, bandpass_response_high,
  spectral_energy_shift_high, neutral
- Strength derivation: spectral energy ratio
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


def _spectral_energy_ratio(embeddings: Tensor) -> float:
    """Compute spectral energy ratio from embeddings."""
    if embeddings.numel() == 0:
        return 0.0
    norms = embeddings.norm(dim=-1)
    return norms.var().item()  # type: ignore[no-any-return]


def _derive_signal(
    ratio: float, thresholds: dict[str, float] | None = None,
) -> tuple[str, EvidenceStrength]:
    """Map spectral energy ratio to detector signal and strength."""
    t = thresholds or {}
    if ratio >= t.get("signal_strong", 0.6):
        return "high_frequency_response_high", "strong"
    if ratio >= t.get("signal_moderate", 0.3):
        return "bandpass_response_high", "moderate"
    if ratio >= t.get("signal_weak", 0.1):
        return "spectral_energy_shift_high", "weak"
    return "neutral", "weak"


def _build_counter_signal(
    ratio: float, thresholds: dict[str, float] | None = None,
) -> str:
    """Build counter-evidence string from spectral analysis."""
    t = thresholds or {}
    if ratio < t.get("counter_low", 0.3):
        return "low_spectral_energy_dominance"
    if ratio < t.get("counter_moderate", 0.5):
        return "moderate_spectral_energy_dominance"
    return "high_spectral_energy_dominance"


class BWGNNAdapter(EvidenceAdapter):
    """Adapter for BWGNN detector with spectral analysis signals."""

    detector_name: str = "bwgnn"

    def __init__(
        self,
        detector: Any,
        graph: Any,
        logits: Tensor,
        spectral_responses: Tensor | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._detector = detector
        self._graph = graph
        self._logits = logits
        self._spectral_responses = spectral_responses
        self._thresholds = thresholds or {}

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
        sr = self._spectral_responses
        return sr is not None and sr.numel() > 0

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

        sr = self._spectral_responses
        if self.supports_detector_signal() and sr is not None and node_id < sr.shape[0]:
            ratio = _spectral_energy_ratio(sr[node_id])
            detector_signal, strength = _derive_signal(ratio, self._thresholds)
            counter_signal = _build_counter_signal(ratio, self._thresholds)
        else:
            detector_signal = "unavailable"
            strength = "unavailable"
            counter_signal = "no_spectral_data"

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

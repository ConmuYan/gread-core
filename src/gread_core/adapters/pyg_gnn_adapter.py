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
    """Build sparse adjacency matrix from edge_index."""
    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
    return torch.sparse_coo_tensor(edge_index, values, (n, n))


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


def _derive_signal(
    cos_dist: float, thresholds: dict[str, float] | None = None,
) -> tuple[str, EvidenceStrength]:
    """Map cosine distance to detector signal and strength."""
    t = thresholds or {}
    if cos_dist >= t.get("signal_strong", 0.4):
        return "embedding_neighbor_discrepancy_high", "strong"
    if cos_dist >= t.get("signal_moderate", 0.25):
        return "message_disagreement_high", "moderate"
    if cos_dist >= t.get("signal_weak", 0.10):
        return "attention_concentration_high", "weak"
    return "neutral", "weak"


def _build_counter_signal(
    cos_dist: float, thresholds: dict[str, float] | None = None,
) -> str:
    """Build counter-evidence string from embedding analysis."""
    t = thresholds or {}
    if cos_dist < t.get("counter_alignment", 0.1):
        return "embedding_neighbor_alignment"
    if cos_dist < t.get("counter_moderate", 0.25):
        return "moderate_embedding_alignment"
    return "low_embedding_alignment"


def _native_metric(signal_family: str, values: Tensor) -> float:
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return 0.0
    if signal_family == "gpr_gnn":
        total = flat.abs().sum().clamp(min=1e-8)
        negative_mass = flat[flat < 0].abs().sum() / total
        concentration = flat.abs().max() / total
        return max(negative_mass.item(), concentration.item())
    if signal_family in {"gin", "h2gcn"}:
        mean_value = flat.abs().mean()
        return (mean_value / (mean_value + 1.0)).item()
    return flat.clamp(0.0, 1.0).mean().item()


def _derive_native_signal(
    signal_family: str,
    value: float,
    thresholds: dict[str, float] | None = None,
) -> tuple[str, EvidenceStrength]:
    t = thresholds or {}
    if signal_family == "pc_gnn":
        if value >= t.get("pc_signal_strong", 0.70):
            return "camouflage_neighbor_filter_high", "strong"
        if value >= t.get("pc_signal_moderate", 0.55):
            return "neighbor_selection_disagreement_high", "moderate"
        if value >= t.get("pc_signal_weak", 0.40):
            return "relation_aware_camouflage_signal", "weak"
        return "neutral", "weak"
    if signal_family == "gpr_gnn":
        if value >= t.get("gpr_signal_strong", 0.50):
            return "high_frequency_response_high", "strong"
        if value >= t.get("gpr_signal_moderate", 0.30):
            return "bandpass_response_high", "moderate"
        if value >= t.get("gpr_signal_weak", 0.10):
            return "spectral_energy_shift_high", "weak"
        return "neutral", "weak"
    if signal_family in {"gin", "h2gcn"}:
        if value >= t.get(f"{signal_family}_signal_strong", 0.50):
            return "relation_anomaly_high", "strong"
        if value >= t.get(f"{signal_family}_signal_moderate", 0.30):
            return "burst_anomaly_high", "moderate"
        if value >= t.get(f"{signal_family}_signal_weak", 0.10):
            return "relation_anomaly_high", "weak"
        return "neutral", "weak"
    return _derive_signal(value, thresholds)


def _build_native_counter_signal(
    signal_family: str,
    value: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    t = thresholds or {}
    if signal_family == "pc_gnn":
        if value < t.get("pc_counter_consistent", 0.40):
            return "consistent_neighbor_selection"
        if value < t.get("pc_counter_moderate", 0.60):
            return "moderate_neighbor_agreement"
        return "high_neighbor_agreement"
    if signal_family == "gpr_gnn":
        if value < t.get("gpr_counter_low", 0.30):
            return "low_spectral_energy_dominance"
        if value < t.get("gpr_counter_moderate", 0.50):
            return "moderate_spectral_energy_dominance"
        return "high_spectral_energy_dominance"
    if value < t.get(f"{signal_family}_counter_low", 0.30):
        return "embedding_neighbor_alignment"
    if value < t.get(f"{signal_family}_counter_moderate", 0.50):
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
        thresholds: dict[str, float] | None = None,
        native_values: Tensor | None = None,
        signal_family: str = "embedding",
        detector_name: str | None = None,
    ) -> None:
        if detector_name is not None:
            self.detector_name = detector_name
        self._detector = detector
        self._graph = graph
        self._logits = logits
        self._embeddings = embeddings
        self._native_values = native_values
        self._signal_family = signal_family
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

    def _get_neighbors(self, node_id: int) -> list[int]:
        mask = self._graph.edge_index[0] == node_id
        return self._graph.edge_index[1][mask].tolist()  # type: ignore[no-any-return]

    def supports_detector_signal(self) -> bool:
        native = self._native_values
        if native is not None and native.numel() > 0:
            return True
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

        native = self._native_values
        has_native = (
            native is not None
            and native.numel() > 0
            and self._signal_family != "embedding"
            and node_id < native.shape[0]
        )
        emb = self._embeddings
        has_emb = emb is not None and emb.numel() > 0 and node_id < emb.shape[0]
        if has_native:
            assert native is not None
            value = _native_metric(self._signal_family, native[node_id])
            detector_signal, strength = _derive_native_signal(
                self._signal_family, value, self._thresholds
            )
            counter_signal = _build_native_counter_signal(
                self._signal_family, value, self._thresholds
            )
        elif has_emb:
            assert emb is not None
            neighbors = self._get_neighbors(node_id)
            if neighbors:
                neighbor_embs = emb[neighbors]  # type: ignore[index]
            else:
                h = emb.shape[-1]  # type: ignore[union-attr]
                neighbor_embs = torch.empty(0, h, dtype=emb.dtype, device=emb.device)
            cos_dist = _embedding_cosine_distance(
                emb[node_id], neighbor_embs  # type: ignore[index]
            )
            detector_signal, strength = _derive_signal(cos_dist, self._thresholds)
            counter_signal = _build_counter_signal(cos_dist, self._thresholds)
        else:
            detector_signal = "unavailable"
            strength = "unavailable"
            counter_signal = "no_detector_native_data"

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

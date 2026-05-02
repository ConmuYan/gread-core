"""GReaDInferencePipeline: LLM-free inference pipeline for graph fraud detection.

Combines a base detector, evidence adapter, and evidence-conditioned reasoner
to produce predictions with risk types, evidence masks, and template explanations.

Tensor shapes:
    z_v:                [B, H]   (node embedding from base detector)
    base_logit:         [B]      (base detector logit)
    evidence_token_ids: [B, K]   (K = number of evidence slots)
    fraud_score:        scalar   (sigmoid of final_logit)

This module is fully LLM-free.  The ``no_llm_guard`` script enforces that
no forbidden network client modules are imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from gread_core.adapters.base import EvidenceAdapter
from gread_core.inference.explanation_template import generate_explanation
from gread_core.models.reasoner import GReaDReasoner
from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.schemas.risk_taxonomy import RISK_TYPES

# Sorted risk type list for deterministic index-to-label mapping.
_RISK_TYPE_LIST: list[str] = sorted(RISK_TYPES)

# Threshold for evidence mask logits to decide supporting vs counter.
_EVIDENCE_MASK_THRESHOLD: float = 0.5


@dataclass
class PredictionResult:
    """Per-node prediction output from the inference pipeline.

    Attributes:
        node_id: Identifier of the node.
        fraud_score: Calibrated fraud probability in [0, 1].
        risk_type: Predicted risk type label.
        supporting_evidence: Evidence slot names with positive mask > threshold.
        counter_evidence: Evidence slot names with negative mask > threshold.
        explanation: Deterministic template-based explanation string.
    """

    node_id: str
    fraud_score: float
    risk_type: str
    supporting_evidence: list[str] = field(default_factory=list)
    counter_evidence: list[str] = field(default_factory=list)
    explanation: str = ""


class GReaDInferencePipeline:
    """LLM-free inference pipeline for GReaD-Core graph fraud detection.

    Orchestrates the base detector, evidence adapter, and evidence-conditioned
    reasoner to produce per-node predictions with risk types, signed evidence
    masks, and deterministic template explanations.

    Args:
        detector: Base graph neural network detector (nn.Module).  Must expose
            ``forward_with_embedding(graph) -> (base_logit, embedding)``.
        reasoner: Trained GReaDReasoner instance.
        adapter: EvidenceAdapter that extracts MinimalEvidencePackage per node.
        config: Configuration dictionary.  Recognised keys:
            - ``evidence.num_slots`` (int): number of evidence slots K.
            - ``evidence.evidence_slot_names`` (list[str]): ordered slot names
              for mapping mask positions back to human-readable labels.
            - ``method.residual_rho`` (float): residual scaling factor.
    """

    def __init__(
        self,
        detector: nn.Module,
        reasoner: GReaDReasoner,
        adapter: EvidenceAdapter,
        config: dict[str, Any],
    ) -> None:
        self.detector = detector
        self.reasoner = reasoner
        self.adapter = adapter
        self.config = config

        self.num_slots: int = config.get("evidence", {}).get("num_slots", 32)
        self.slot_names: list[str] = config.get("evidence", {}).get(
            "evidence_slot_names",
            [f"slot_{i}" for i in range(self.num_slots)],
        )

    # ------------------------------------------------------------------
    # Evidence encoding
    # ------------------------------------------------------------------

    def _encode_evidence(
        self,
        meps: list[MinimalEvidencePackage],
    ) -> Tensor:
        """Convert a list of MinimalEvidencePackage objects into token IDs.

        Each evidence field present in the MEP's ReasoningChannel is mapped to
        a discrete token ID (its position index in ``slot_names``).  Slots not
        present in the MEP receive token ID 0 (padding).

        Args:
            meps: One MinimalEvidencePackage per node.

        Returns:
            LongTensor of shape [B, K].
        """
        slot_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.slot_names)
        }

        batch_ids: list[list[int]] = []
        for mep in meps:
            ids = [0] * self.num_slots
            reasoning = mep.reasoning
            # Map each reasoning-channel field to its slot index.
            evidence_fields: list[str] = [
                reasoning.uncertainty_level,
                reasoning.degree_level,
                reasoning.neighbor_consistency,
                reasoning.feature_neighbor_discrepancy,
                reasoning.detector_signal,
                reasoning.detector_signal_strength,
                reasoning.counter_signal,
            ]
            for field_name in evidence_fields:
                if field_name in slot_to_idx:
                    ids[slot_to_idx[field_name]] = slot_to_idx[field_name]
            batch_ids.append(ids)

        return torch.tensor(batch_ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        graph: object,
        node_ids: list[int],
    ) -> list[PredictionResult]:
        """Run LLM-free inference on the specified nodes.

        Steps:
            1. Call ``detector.forward_with_embedding(graph)`` to obtain
               ``base_logit`` [B] and ``embeddings`` [B, H].
            2. Use ``adapter.extract(node_ids)`` to get
               ``list[MinimalEvidencePackage]``.
            3. Encode evidence packages into ``evidence_token_ids`` [B, K].
            4. Call ``reasoner.forward(z_v, base_logit, evidence_token_ids)``
               to obtain ``final_logit``, ``type_logits``, and mask logits.
            5. Map ``type_logits`` to ``risk_type`` via ``RISK_TYPES``.
            6. Threshold ``pos_mask_logits`` / ``neg_mask_logits`` at 0.5 to
               determine supporting / counter evidence slot names.
            7. Generate a deterministic template explanation.
            8. Return a list of ``PredictionResult`` dataclasses.

        Args:
            graph: PyG Data object consumed by the base detector.
            node_ids: List of node indices to produce predictions for.

        Returns:
            List of ``PredictionResult``, one per node.

        Raises:
            ValueError: If ``node_ids`` is empty.
        """
        if not node_ids:
            raise ValueError("node_ids must be non-empty")

        self.detector.eval()
        self.reasoner.eval()

        # Step 1: detector forward pass.
        with torch.no_grad():
            base_logit, embeddings = self.detector.forward_with_embedding(graph)  # type: ignore[operator]
        # embeddings: [B, H],  base_logit: [B]

        # Step 2: extract evidence packages.
        meps = self.adapter.extract(node_ids)

        # Step 3: encode evidence to token IDs.
        evidence_token_ids = self._encode_evidence(meps)  # [B, K]
        # Move to same device as embeddings.
        evidence_token_ids = evidence_token_ids.to(embeddings.device)

        # Step 4: reasoner forward pass.
        with torch.no_grad():
            outputs = self.reasoner.forward(
                z_v=embeddings,
                base_logit=base_logit,
                evidence_token_ids=evidence_token_ids,
            )

        final_logit: Tensor = outputs["final_logit"]  # [B]
        type_logits: Tensor = outputs["type_logits"]  # [B, T]
        pos_mask_logits: Tensor = outputs["pos_mask_logits"]  # [B, K]
        neg_mask_logits: Tensor = outputs["neg_mask_logits"]  # [B, K]

        # Step 5-8: assemble per-node results.
        fraud_scores = torch.sigmoid(final_logit).cpu()  # [B]
        type_indices = type_logits.argmax(dim=-1).cpu()  # [B]
        pos_mask = (pos_mask_logits > _EVIDENCE_MASK_THRESHOLD).cpu()  # [B, K]
        neg_mask = (neg_mask_logits > _EVIDENCE_MASK_THRESHOLD).cpu()  # [B, K]

        results: list[PredictionResult] = []
        for i, mep in enumerate(meps):
            # Risk type.
            risk_idx = int(type_indices[i].item())
            risk_type = (
                _RISK_TYPE_LIST[risk_idx]
                if risk_idx < len(_RISK_TYPE_LIST)
                else "weak_or_uncertain_evidence"
            )

            # Supporting / counter evidence slot names.
            supporting = [
                self.slot_names[j]
                for j in range(self.num_slots)
                if pos_mask[i, j].item()
            ]
            counter = [
                self.slot_names[j]
                for j in range(self.num_slots)
                if neg_mask[i, j].item()
            ]

            # Deterministic template explanation.
            explanation = generate_explanation(
                risk_type=risk_type,
                supporting_evidence=supporting,
                counter_evidence=counter,
            )

            results.append(
                PredictionResult(
                    node_id=str(mep.node_id),
                    fraud_score=float(fraud_scores[i].item()),
                    risk_type=risk_type,
                    supporting_evidence=supporting,
                    counter_evidence=counter,
                    explanation=explanation,
                )
            )

        return results

"""GReaDReasoner: orchestrator for the evidence-conditioned student reasoner.

Combines EvidenceEncoder, RiskTypeHead, SignedEvidenceHead, and
EvidenceGatedResidualReadout into a single forward pass.

Tensor shapes:
    z_v:                 [B, H]    (node embedding from base detector)
    base_logit:          [B]       (base detector logit)
    evidence_token_ids:  [B, K]    (K = number of evidence slots)
    final_logit:         [B]       (base_logit + rho * residual_logit)
    type_logits:         [B, T]    (T = number of risk types)
    pos_mask_logits:     [B, K]
    neg_mask_logits:     [B, K]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.heads import RiskTypeHead, SignedEvidenceHead
from gread_core.models.residual_readout import EvidenceGatedResidualReadout


class GReaDReasoner(nn.Module):
    """Evidence-conditioned residual reasoner for graph fraud detection.

    Orchestrates the evidence encoder, prediction heads, and residual readout
    to produce fraud scores, risk type predictions, and signed evidence masks.

    The final fraud logit is:
        final_logit = base_logit + rho * residual_logit

    When rho=0, final_logit equals base_logit exactly (ablation baseline).

    Args:
        hidden_dim: Dimension of base detector node embeddings (H).
        evidence_encoder: Module that maps evidence_token_ids [B, K] -> [B, E].
        num_risk_types: Number of risk type classes (T).
        num_evidence_slots: Number of evidence slots (K).
        rho: Residual scaling factor. Default 0.1.
    """

    def __init__(
        self,
        hidden_dim: int,
        evidence_encoder: EvidenceEncoder,
        num_risk_types: int,
        num_evidence_slots: int,
        rho: float = 0.1,
    ) -> None:
        super().__init__()
        self.evidence_encoder = evidence_encoder
        self.rho = rho

        evidence_dim = evidence_encoder.output_dim
        combined_dim = hidden_dim + evidence_dim

        self.type_head = RiskTypeHead(combined_dim, num_risk_types)
        self.signed_evidence_head = SignedEvidenceHead(combined_dim, num_evidence_slots)
        self.residual_readout = EvidenceGatedResidualReadout(hidden_dim, evidence_dim)

    def forward(
        self,
        z_v: Tensor,
        base_logit: Tensor,
        evidence_token_ids: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            z_v: [B, H] node embeddings from base detector.
            base_logit: [B] base detector classification logits.
            evidence_token_ids: [B, K] LongTensor of evidence token IDs.

        Returns:
            Dict with keys:
                - base_logit: [B] original detector logits
                - final_logit: [B] base_logit + rho * residual_logit
                - type_logits: [B, T] risk type logits
                - pos_mask_logits: [B, K] positive evidence mask logits
                - neg_mask_logits: [B, K] negative evidence mask logits
        """
        g_v = self.evidence_encoder(evidence_token_ids)  # [B, E]
        h = torch.cat([z_v, g_v], dim=-1)  # [B, H+E]

        type_logits = self.type_head(h)  # [B, T]
        pos_mask_logits, neg_mask_logits = self.signed_evidence_head(h)  # [B, K], [B, K]

        residual_logit = self.residual_readout(
            z_v=z_v,
            evidence_embedding=g_v,
            pos_mask_logits=pos_mask_logits,
            neg_mask_logits=neg_mask_logits,
        )  # [B]

        final_logit = base_logit + self.rho * residual_logit  # [B]

        return {
            "base_logit": base_logit,
            "final_logit": final_logit,
            "type_logits": type_logits,
            "pos_mask_logits": pos_mask_logits,
            "neg_mask_logits": neg_mask_logits,
        }

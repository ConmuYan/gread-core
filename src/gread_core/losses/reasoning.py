"""Reasoning loss: L = L_sup + lambda * a_v * (L_type + L_evidence).

CRITICAL CONTRACT:
- Only a_v=1 (accepted) samples contribute to type/evidence loss.
- Rejected ERRs produce zero type/evidence loss.
- summary is NEVER used in loss computation.
- DHEF, CER, adaptive lambda are experimental only and disabled by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gread_core.losses.supervised import supervised_loss


class ReasoningLoss(nn.Module):
    """Combined loss for the evidence-conditioned reasoner.

    L = L_sup + lambda_reason * accepted_mask * (L_type + L_evidence)

    If accepted_mask.sum() == 0:
        type_loss = 0, evidence_loss = 0, total_loss = L_sup

    Args:
        lambda_reason: Weight for reasoning losses (type + evidence).
        num_risk_types: Number of risk type classes for type classification.
    """

    def __init__(
        self,
        lambda_reason: float = 0.5,
        num_risk_types: int = 6,
    ) -> None:
        super().__init__()
        self.lambda_reason = lambda_reason
        self.num_risk_types = num_risk_types
        self.type_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.evidence_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        final_logit: Tensor,
        type_logits: Tensor,
        pos_mask_logits: Tensor,
        neg_mask_logits: Tensor,
        labels: Tensor,
        risk_type_targets: Tensor | None,
        pos_evidence_targets: Tensor | None,
        neg_evidence_targets: Tensor | None,
        accepted_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the combined reasoning loss.

        Args:
            final_logit: [B] final fraud logits from reasoner.
            type_logits: [B, T] risk type logits.
            pos_mask_logits: [B, K] positive evidence mask logits.
            neg_mask_logits: [B, K] negative evidence mask logits.
            labels: [B] binary labels (0=benign, 1=fraud).
            risk_type_targets: [B] risk type class indices (only for accepted).
            pos_evidence_targets: [B, K] positive evidence binary targets.
            neg_evidence_targets: [B, K] negative evidence binary targets.
            accepted_mask: [B] boolean mask, True for accepted ERR samples.

        Returns:
            Dict with keys:
                - total_loss: scalar
                - sup_loss: scalar
                - type_loss: scalar (0 if no accepted samples)
                - evidence_loss: scalar (0 if no accepted samples)
        """
        # L_sup: always computed on all samples
        sup_loss = supervised_loss(final_logit, labels)

        # Type and evidence losses: only for accepted samples
        num_accepted = accepted_mask.sum()

        if num_accepted == 0 or risk_type_targets is None:
            return {
                "total_loss": sup_loss,
                "sup_loss": sup_loss,
                "type_loss": torch.tensor(0.0, device=sup_loss.device),
                "evidence_loss": torch.tensor(0.0, device=sup_loss.device),
            }

        # L_type: cross-entropy, masked by accepted
        type_loss_per_sample = self.type_loss_fn(type_logits, risk_type_targets)
        type_loss = (type_loss_per_sample * accepted_mask.float()).sum() / num_accepted.clamp(min=1)

        # L_evidence: BCE on signed evidence masks, masked by accepted
        evidence_loss = torch.tensor(0.0, device=sup_loss.device)
        if pos_evidence_targets is not None and neg_evidence_targets is not None:
            pos_loss = self.evidence_loss_fn(pos_mask_logits, pos_evidence_targets.float())
            neg_loss = self.evidence_loss_fn(neg_mask_logits, neg_evidence_targets.float())
            evidence_per_sample = (pos_loss + neg_loss).mean(dim=-1)  # [B]
            evidence_loss = (
                (evidence_per_sample * accepted_mask.float()).sum()
                / num_accepted.clamp(min=1)
            )

        total_loss = sup_loss + self.lambda_reason * (type_loss + evidence_loss)

        return {
            "total_loss": total_loss,
            "sup_loss": sup_loss,
            "type_loss": type_loss,
            "evidence_loss": evidence_loss,
        }

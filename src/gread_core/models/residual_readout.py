"""EvidenceGatedResidualReadout: compute a residual logit adjustment gated by evidence.

The residual readout computes an MLP-based adjustment to the base detector logit,
modulated by the net direction of signed evidence masks.

Tensor shapes:
    z_v:                 [B, H]    (node embedding)
    evidence_embedding:  [B, E]    (evidence embedding g_v)
    pos_mask_logits:     [B, K]    (positive evidence logits)
    neg_mask_logits:     [B, K]    (negative evidence logits)
    output:              [B]       (residual logit)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EvidenceGatedResidualReadout(nn.Module):
    """Compute evidence-gated residual logit adjustment.

    The module:
    1. Concatenates [z_v; g_v] and passes through an MLP to get a raw residual.
    2. Computes a net evidence gate from the difference of mean sigmoid
       activations of positive and negative mask logits.
    3. Returns raw_residual * net_gate as the final residual logit.

    When all evidence masks are zero (or rho=0 is applied externally),
    the residual contribution vanishes, recovering the base detector logit.

    Args:
        hidden_dim: Dimension of node embeddings (H).
        evidence_dim: Dimension of evidence embeddings (E).
    """

    def __init__(self, hidden_dim: int, evidence_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + evidence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_v: Tensor,
        evidence_embedding: Tensor,
        pos_mask_logits: Tensor,
        neg_mask_logits: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            z_v: [B, H] node embeddings.
            evidence_embedding: [B, E] evidence embeddings.
            pos_mask_logits: [B, K] positive evidence mask logits.
            neg_mask_logits: [B, K] negative evidence mask logits.

        Returns:
            [B] residual logit.
        """
        h: Tensor = torch.cat([z_v, evidence_embedding], dim=-1)  # [B, H+E]
        raw_residual: Tensor = self.mlp(h).squeeze(-1)  # [B]

        pos_gate: Tensor = torch.sigmoid(pos_mask_logits).mean(dim=1)  # [B]
        neg_gate: Tensor = torch.sigmoid(neg_mask_logits).mean(dim=1)  # [B]
        net_gate: Tensor = pos_gate - neg_gate  # [B]

        result: Tensor = raw_residual * net_gate  # [B]
        return result

"""Prediction heads for the student reasoner.

RiskTypeHead: classifies risk type from concatenated [z_v; g_v].
SignedEvidenceHead: independent positive and negative evidence mask heads.

Tensor shapes:
    input h:             [B, D]    (D = hidden_dim + evidence_dim)
    type_logits:         [B, T]    (T = num_risk_types)
    pos_mask_logits:     [B, K]    (K = num_evidence_slots)
    neg_mask_logits:     [B, K]
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class RiskTypeHead(nn.Module):
    """Linear projection for risk type classification.

    Args:
        input_dim: Dimension of input features (D).
        num_risk_types: Number of risk types (T).
    """

    def __init__(self, input_dim: int, num_risk_types: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_risk_types)

    def forward(self, h: Tensor) -> Tensor:
        """Forward pass.

        Args:
            h: [B, D] concatenated node and evidence embeddings.

        Returns:
            [B, T] risk type logits.
        """
        result: Tensor = self.fc(h)
        return result


class SignedEvidenceHead(nn.Module):
    """Independent positive and negative evidence mask heads.

    The two heads share no parameters, allowing the model to learn
    distinct patterns for supporting vs. counter evidence.

    Args:
        input_dim: Dimension of input features (D).
        num_evidence_slots: Number of evidence slots (K).
    """

    def __init__(self, input_dim: int, num_evidence_slots: int) -> None:
        super().__init__()
        self.pos_head = nn.Linear(input_dim, num_evidence_slots)
        self.neg_head = nn.Linear(input_dim, num_evidence_slots)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            h: [B, D] concatenated node and evidence embeddings.

        Returns:
            Tuple of (pos_mask_logits [B, K], neg_mask_logits [B, K]).
        """
        return self.pos_head(h), self.neg_head(h)

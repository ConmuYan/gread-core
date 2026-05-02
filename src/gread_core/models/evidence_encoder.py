"""EvidenceEncoder: embed discrete evidence token IDs into dense vectors.

Tensor shapes:
    evidence_token_ids:  [B, K]    (K = number of evidence slots, LongTensor)
    output:              [B, E]    (E = output_dim)
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class EvidenceEncoder(nn.Module):
    """Embed discrete evidence token IDs and project to a dense vector.

    Args:
        vocab_size: Number of distinct evidence token IDs.
        embed_dim: Dimension of each token embedding.
        num_slots: Number of evidence slots per sample (K).
        output_dim: Dimension of the output evidence embedding (E).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_slots: int,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.num_slots = num_slots
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * num_slots, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, evidence_token_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            evidence_token_ids: [B, K] LongTensor of evidence token IDs.

        Returns:
            [B, output_dim] evidence embedding.
        """
        x: Tensor = self.embedding(evidence_token_ids)  # [B, K, embed_dim]
        x = x.flatten(start_dim=1)  # [B, K * embed_dim]
        result: Tensor = self.proj(x)  # [B, output_dim]
        return result

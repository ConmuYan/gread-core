"""Supervised loss: L_sup = BCEWithLogitsLoss for fraud classification."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def supervised_loss(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Binary cross-entropy with logits for fraud detection.

    Args:
        logits: [B] raw classification logits.
        labels: [B] binary labels (0=benign, 1=fraud).
        mask: [B] optional boolean mask for selecting which samples to include.

    Returns:
        Scalar loss tensor.
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    if mask is not None:
        loss = loss * mask.float()
        return loss.sum() / mask.float().sum().clamp(min=1.0)
    return loss.mean()

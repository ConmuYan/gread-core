"""Stage 1: Base detector warm-up training.

CRITICAL CONSTRAINT: Stage 1 must NOT import LLM.
This stage trains only the base detector (GCN/GAT) on supervised classification.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

from gread_core.losses.supervised import supervised_loss
from gread_core.training.checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


def _make_mask_view(data: Data, active_mask: str) -> Data:
    """Create a shallow copy of data with only the requested mask active.

    forward_with_embedding picks test > val > train. To force it to use
    a specific mask, we hide the higher-priority masks by setting them to None.
    """
    view = data.clone()

    # Hide higher-priority masks
    if active_mask == "train":
        view.test_mask = None
        view.val_mask = None
    elif active_mask == "val":
        view.test_mask = None

    return view


def train_detector(
    detector: Any,
    data: Data,
    config: dict[str, Any],
    checkpoint_manager: CheckpointManager | None = None,
    writer: Any = None,
) -> Any:
    """Stage 1: Train base detector with supervised loss.

    Args:
        detector: The base detector model (must implement DetectorProtocol).
        data: PyG Data object with x, edge_index, y, train_mask, val_mask.
        config: Training configuration dict.
        checkpoint_manager: Optional checkpoint manager for saving.

    Returns:
        Trained detector model.
    """
    stage_cfg = config.get("stage1", {})
    epochs = stage_cfg.get("epochs", 100)
    lr = stage_cfg.get("lr", 0.01)
    weight_decay = stage_cfg.get("weight_decay", 5e-4)
    log_every = stage_cfg.get("log_every", 10)

    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)

    # Create data views for train/val so the detector picks the right mask.
    # forward_with_embedding picks test > val > train, so we hide higher masks.
    train_data = _make_mask_view(data, active_mask="train")
    val_data = _make_mask_view(data, active_mask="val")
    train_labels = data.y[data.train_mask]
    val_labels = data.y[data.val_mask]

    detector.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        logits, _ = detector.forward_with_embedding(train_data)
        loss = supervised_loss(logits, train_labels)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            with torch.no_grad():
                preds = (logits > 0).long()
                acc = (preds == train_labels).float().mean().item()
            logger.info(
                "Stage 1 | Epoch %d/%d | Loss: %.4f | Train Acc: %.4f",
                epoch, epochs, loss.item(), acc,
            )
            if writer is not None:
                writer.add_scalar("stage1/train_loss", loss.item(), epoch)
                writer.add_scalar("stage1/train_acc", acc, epoch)

        # Validation
        if epoch % log_every == 0:
            detector.eval()
            with torch.no_grad():
                val_logits, _ = detector.forward_with_embedding(val_data)
                val_loss = supervised_loss(val_logits, val_labels)
                val_preds = (val_logits > 0).long()
                val_acc = (val_preds == val_labels).float().mean().item()
            logger.info(
                "Stage 1 | Epoch %d | Val Loss: %.4f | Val Acc: %.4f",
                epoch, val_loss.item(), val_acc,
            )
            if writer is not None:
                writer.add_scalar("stage1/val_loss", val_loss.item(), epoch)
                writer.add_scalar("stage1/val_acc", val_acc, epoch)
            detector.train()

        # Checkpoint
        if checkpoint_manager is not None and epoch % stage_cfg.get("save_every", epochs) == 0:
            checkpoint_manager.save(detector, stage=1, epoch=epoch, optimizer=optimizer)

    # Final checkpoint
    if checkpoint_manager is not None:
        checkpoint_manager.save(detector, stage=1, epoch=epochs, optimizer=optimizer)

    return detector

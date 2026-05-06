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

from gread_core.evaluation.detection import compute_all_detection_metrics
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


def _summarize_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    scores = torch.sigmoid(logits)
    preds = (scores >= 0.5).long()
    metrics = compute_all_detection_metrics(
        labels.detach().cpu().numpy(),
        scores.detach().cpu().numpy(),
        preds.detach().cpu().numpy(),
    )
    metrics["accuracy"] = float((preds == labels).float().mean().item())
    return metrics


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
    epochs = stage_cfg.get("epochs", 500)
    lr = stage_cfg.get("lr", 0.01)
    weight_decay = stage_cfg.get("weight_decay", 5e-4)
    log_every = stage_cfg.get("log_every", 10)
    save_every = stage_cfg.get("save_every", epochs)
    early_stop_cfg = stage_cfg.get("early_stopping", {})
    early_stop_enabled = early_stop_cfg.get("enabled", True)
    monitor = str(early_stop_cfg.get("monitor", "auprc"))
    patience = int(early_stop_cfg.get("patience", 50))
    min_delta = float(early_stop_cfg.get("min_delta", 1e-4))
    warmup_epochs = int(early_stop_cfg.get("warmup_epochs", 0))

    optimizer = torch.optim.Adam(detector.parameters(), lr=lr, weight_decay=weight_decay)

    # Create data views for train/val so the detector picks the right mask.
    # forward_with_embedding picks test > val > train, so we hide higher masks.
    train_data = _make_mask_view(data, active_mask="train")
    val_data = _make_mask_view(data, active_mask="val")
    train_labels = data.y[data.train_mask]
    val_labels = data.y[data.val_mask]
    best_metric = float("-inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in detector.state_dict().items()}
    epochs_without_improvement = 0
    last_epoch = 0
    stopped_early = False

    detector.train()
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        optimizer.zero_grad()

        logits, _ = detector.forward_with_embedding(train_data)
        loss = supervised_loss(logits, train_labels)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        train_metrics = _summarize_logits(logits.detach(), train_labels)

        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                "Stage 1 | Epoch %d/%d | Loss: %.4f | Train Acc: %.4f | Train AUPRC: %.4f",
                epoch, epochs, loss.item(), train_metrics["accuracy"], train_metrics["auprc"],
            )
            if writer is not None:
                writer.add_scalar("stage1/train_loss", loss.item(), epoch)
                writer.add_scalar("stage1/train_acc", train_metrics["accuracy"], epoch)
                writer.add_scalar("stage1/train_auprc", train_metrics["auprc"], epoch)

        detector.eval()
        with torch.no_grad():
            val_logits, _ = detector.forward_with_embedding(val_data)
            val_loss = supervised_loss(val_logits, val_labels)
            val_metrics = _summarize_logits(val_logits, val_labels)

        current_metric = float(val_metrics.get(monitor, val_metrics["auprc"]))
        improved = current_metric > (best_metric + min_delta)
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in detector.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % log_every == 0 or epoch == 1 or improved:
            logger.info(
                (
                    "Stage 1 | Epoch %d | Val Loss: %.4f | Val AUROC: %.4f | "
                    "Val AUPRC: %.4f | Val F1-Macro: %.4f | Val G-Means: %.4f | "
                    "Best %s: %.4f @ %d"
                ),
                epoch,
                val_loss.item(),
                val_metrics["auroc"],
                val_metrics["auprc"],
                val_metrics["f1_macro"],
                val_metrics["g_means"],
                monitor,
                best_metric,
                best_epoch,
            )
            if writer is not None:
                writer.add_scalar("stage1/val_loss", val_loss.item(), epoch)
                writer.add_scalar("stage1/val_acc", val_metrics["accuracy"], epoch)
                writer.add_scalar("stage1/val_auroc", val_metrics["auroc"], epoch)
                writer.add_scalar("stage1/val_auprc", val_metrics["auprc"], epoch)
                writer.add_scalar("stage1/val_f1_macro", val_metrics["f1_macro"], epoch)
                writer.add_scalar("stage1/val_g_means", val_metrics["g_means"], epoch)
                writer.add_scalar(f"stage1/val_monitor_{monitor}", current_metric, epoch)
        detector.train()

        if checkpoint_manager is not None and epoch % save_every == 0:
            checkpoint_manager.save(
                detector,
                stage=1,
                epoch=epoch,
                optimizer=optimizer,
                extra={
                    "is_best": improved,
                    "monitor": monitor,
                    "monitor_value": current_metric,
                    "val_metrics": val_metrics,
                },
            )

        if (
            early_stop_enabled
            and epoch >= warmup_epochs
            and patience > 0
            and epochs_without_improvement >= patience
        ):
            stopped_early = True
            logger.info(
                (
                    "Stage 1 early stop at epoch %d after %d stale validation checks. "
                    "Best %s: %.4f @ %d"
                ),
                epoch,
                epochs_without_improvement,
                monitor,
                best_metric,
                best_epoch,
            )
            break

    detector.load_state_dict(best_state)
    detector.stage1_training_info = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "monitor": monitor,
        "stopped_early": stopped_early,
        "last_epoch": last_epoch,
    }

    if checkpoint_manager is not None:
        checkpoint_manager.save(
            detector,
            stage=1,
            epoch=last_epoch,
            optimizer=optimizer,
            extra={
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "monitor": monitor,
                "stopped_early": stopped_early,
                "saved_model_epoch": best_epoch,
            },
        )

    return detector

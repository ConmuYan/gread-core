"""Stage 3: Reasoner distillation using accepted ERRs.

CRITICAL CONSTRAINTS:
- Stage 3 uses accepted ERR only (rejected excluded from reasoning loss).
- Rejected ERRs produce zero type/evidence loss.
- summary is NEVER used in loss computation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

from gread_core.losses.reasoning import ReasoningLoss
from gread_core.models.reasoner import GReaDReasoner
from gread_core.schemas.risk_taxonomy import (
    EVIDENCE_SLOT_TO_INDEX,
    RISK_TYPE_TO_INDEX,
    encode_evidence_slots,
)
from gread_core.training.checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


def _strip_masks(data: Data) -> Data:
    """Create a shallow copy of data with all masks removed.

    This ensures forward_with_embedding returns embeddings for all nodes.
    """
    view = data.clone()
    view.train_mask = None
    view.val_mask = None
    view.test_mask = None
    return view


def _build_training_batch(
    accepted_errs: list[dict[str, Any]],
    node_embeddings: Tensor,
    base_logits: Tensor,
    num_evidence_slots: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Build training batch from accepted ERRs and detector outputs.

    Only accepted ERRs are included. summary is NEVER used.

    Args:
        accepted_errs: List of accepted ERR dicts from Stage 2.
        node_embeddings: [N, H] node embeddings from detector.
        base_logits: [N] base detector logits.
        num_evidence_slots: K (number of evidence slots).
        device: Target device.

    Returns:
        Dict with batched tensors for training.
    """
    if not accepted_errs:
        return {}

    batch_indices = []
    risk_types = []
    pos_evidence = []
    neg_evidence = []
    evidence_token_ids_list = []

    for err_dict in accepted_errs:
        idx = err_dict["node_idx"]
        err = err_dict["err"]

        batch_indices.append(idx)
        risk_types.append(RISK_TYPE_TO_INDEX.get(err["risk_type"], 0))

        # Build evidence target vectors
        support_ids = err.get("supporting_evidence", [])
        counter_ids = err.get("counter_evidence", [])

        pos_vec = _evidence_ids_to_vector(support_ids, num_evidence_slots)
        neg_vec = _evidence_ids_to_vector(counter_ids, num_evidence_slots)

        pos_evidence.append(pos_vec)
        neg_evidence.append(neg_vec)

        # Build evidence token IDs from all cited evidence fields
        all_evidence_ids = support_ids + counter_ids
        token_ids = _encode_evidence_to_tokens(all_evidence_ids, num_evidence_slots)
        evidence_token_ids_list.append(token_ids)

    indices = torch.tensor(batch_indices, dtype=torch.long, device=device)

    return {
        "node_indices": indices,
        "z_v": node_embeddings[indices],
        "base_logit": base_logits[indices],
        "risk_type_targets": torch.tensor(risk_types, dtype=torch.long, device=device),
        "pos_evidence_targets": torch.stack(pos_evidence).to(device),
        "neg_evidence_targets": torch.stack(neg_evidence).to(device),
        "evidence_token_ids": torch.stack(evidence_token_ids_list).to(device),
    }


def _encode_evidence_to_tokens(evidence_ids: list[str], num_slots: int) -> Tensor:
    """Convert evidence ID list to token IDs for the evidence encoder."""
    return torch.tensor(encode_evidence_slots(evidence_ids, num_slots), dtype=torch.long)


def _evidence_ids_to_vector(evidence_ids: list[str], num_slots: int) -> Tensor:
    """Convert evidence ID list to binary vector.

    Maps evidence IDs to slot indices using canonical EVIDENCE_SLOT_TO_INDEX.
    Unknown IDs are ignored.
    """
    vec = torch.zeros(num_slots)
    for eid in evidence_ids:
        if eid in EVIDENCE_SLOT_TO_INDEX:
            slot = EVIDENCE_SLOT_TO_INDEX[eid]
            if slot < num_slots:
                vec[slot] = 1.0
    return vec


def train_reasoner(
    reasoner: GReaDReasoner,
    detector: Any,
    data: Data,
    accepted_errs: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint_manager: CheckpointManager | None = None,
    device: torch.device | None = None,
    writer: Any = None,
) -> GReaDReasoner:
    """Stage 3: Train the reasoner using accepted ERRs only.

    Args:
        reasoner: The GReaDReasoner model to train.
        detector: Trained base detector (frozen).
        data: PyG Data object.
        accepted_errs: List of accepted ERR dicts from Stage 2.
        config: Configuration dict.
        checkpoint_manager: Optional checkpoint manager.
        device: Target device.

    Returns:
        Trained reasoner model.
    """
    if device is None:
        device = next(reasoner.parameters()).device

    stage_cfg = config.get("stage3", {})
    epochs = stage_cfg.get("epochs", 50)
    lr = stage_cfg.get("lr", 0.001)
    weight_decay = stage_cfg.get("weight_decay", 1e-5)
    lambda_reason = config.get("method", {}).get("lambda_reason", 0.5)
    log_every = stage_cfg.get("log_every", 5)

    num_risk_types = len(RISK_TYPE_TO_INDEX)
    num_evidence_slots = config.get("evidence", {}).get("num_slots", 32)

    # Freeze detector
    detector.eval()
    for param in detector.parameters():
        param.requires_grad = False

    # Get detector outputs for all nodes (remove masks so no filtering)
    no_mask_data = _strip_masks(data)
    with torch.no_grad():
        base_logits, node_embeddings = detector.forward_with_embedding(no_mask_data)
        node_embeddings = node_embeddings.detach().to(device)
        base_logits = base_logits.detach().to(device)

    # Build training batch from accepted ERRs only
    batch = _build_training_batch(
        accepted_errs, node_embeddings, base_logits, num_evidence_slots, device
    )

    if not batch:
        logger.warning("No accepted ERRs available for Stage 3 training")
        return reasoner

    # Loss function
    loss_fn = ReasoningLoss(
        lambda_reason=lambda_reason,
        num_risk_types=num_risk_types,
    )

    optimizer = torch.optim.Adam(reasoner.parameters(), lr=lr, weight_decay=weight_decay)

    # All samples in this batch are accepted (accepted_mask = all True)
    num_samples = len(batch["node_indices"])
    accepted_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
    labels = data.y.to(device)[batch["node_indices"]]

    reasoner.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        outputs = reasoner(
            z_v=batch["z_v"],
            base_logit=batch["base_logit"],
            evidence_token_ids=batch["evidence_token_ids"],
        )

        losses = loss_fn(
            final_logit=outputs["final_logit"],
            type_logits=outputs["type_logits"],
            pos_mask_logits=outputs["pos_mask_logits"],
            neg_mask_logits=outputs["neg_mask_logits"],
            labels=labels,
            risk_type_targets=batch["risk_type_targets"],
            pos_evidence_targets=batch["pos_evidence_targets"],
            neg_evidence_targets=batch["neg_evidence_targets"],
            accepted_mask=accepted_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            logger.info(
                "Stage 3 | Epoch %d/%d | Total: %.4f | Sup: %.4f | Type: %.4f | Evidence: %.4f",
                epoch, epochs,
                losses["total_loss"].item(),
                losses["sup_loss"].item(),
                losses["type_loss"].item(),
                losses["evidence_loss"].item(),
            )
            if writer is not None:
                writer.add_scalar("stage3/total_loss", losses["total_loss"].item(), epoch)
                writer.add_scalar("stage3/sup_loss", losses["sup_loss"].item(), epoch)
                writer.add_scalar("stage3/type_loss", losses["type_loss"].item(), epoch)
                writer.add_scalar("stage3/evidence_loss", losses["evidence_loss"].item(), epoch)

        if checkpoint_manager is not None and epoch % stage_cfg.get("save_every", epochs) == 0:
            checkpoint_manager.save(reasoner, stage=3, epoch=epoch, optimizer=optimizer)

    # Final checkpoint
    if checkpoint_manager is not None:
        checkpoint_manager.save(reasoner, stage=3, epoch=epochs, optimizer=optimizer)

    return reasoner

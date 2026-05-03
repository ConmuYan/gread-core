"""Tri-CEC: Counterfactual Evidence Consistency evaluation.

Three dimensions:
- Score-CEC: weaken evidence -> score should change
- Type-CEC: weaken evidence -> risk type should change
- Evidence-CEC: weaken evidence -> evidence masks should change

CRITICAL: prediction_score is calibration-only and must NOT be modified
by evidence weakening operations.

All operations are deterministic given the same input. No LLM dependencies.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import LongTensor

from gread_core.schemas.evidence import (
    MinimalEvidencePackage,
    ReasoningChannel,
)

# Default weakening config: maps reasoning channel fields to value replacements.
# Each key is a ReasoningChannel field; value maps original -> weakened.
DEFAULT_WEAKEN_CONFIG: dict[str, dict[str, str]] = {
    "detector_signal": {
        "high_frequency_response_high": "neutral_frequency_response_medium",
        "high_frequency_response_medium": "neutral_frequency_response_medium",
        "neutral": "low_frequency_response_low",
        "embedding_neighbor_discrepancy_high": "neutral",
        "message_disagreement_high": "neutral",
        "attention_concentration_high": "neutral",
    },
    "detector_signal_strength": {
        "strong": "weak",
        "moderate": "weak",
        "weak": "unavailable",
    },
    "neighbor_consistency": {
        "low": "high",
        "medium": "high",
        "high": "low",
    },
    "feature_neighbor_discrepancy": {
        "high": "low",
        "medium": "low",
        "low": "high",
    },
    "degree_level": {
        "burst": "normal",
        "high": "normal",
        "medium": "low",
        "low": "high",
        "normal": "burst",
    },
}


def weaken_evidence(
    mep: MinimalEvidencePackage,
    weaken_config: dict[str, dict[str, str]] | None = None,
) -> MinimalEvidencePackage:
    """Create a weakened copy of a MEP by replacing reasoning channel values.

    CRITICAL: Only reasoning channel fields are modified.
    CalibrationChannel (prediction_score, uncertainty) is NEVER touched.

    Args:
        mep: Original MinimalEvidencePackage.
        weaken_config: Mapping of field -> {old_value: new_value}.
            Defaults to DEFAULT_WEAKEN_CONFIG.

    Returns:
        New MinimalEvidencePackage with weakened reasoning fields.
    """
    if weaken_config is None:
        weaken_config = DEFAULT_WEAKEN_CONFIG

    reasoning_dict = mep.reasoning.model_dump()

    for field, replacements in weaken_config.items():
        if field in reasoning_dict:
            current_val = reasoning_dict[field]
            if current_val in replacements:
                reasoning_dict[field] = replacements[current_val]

    return MinimalEvidencePackage(
        node_id=mep.node_id,
        detector_name=mep.detector_name,
        calibration=mep.calibration,  # NEVER modified
        reasoning=ReasoningChannel(**reasoning_dict),
    )


def _mep_to_evidence_token_ids(
    mep: MinimalEvidencePackage,
    slot_to_id: dict[str, int],
    num_slots: int,
    device: torch.device | None = None,
) -> LongTensor:
    """Convert MEP reasoning fields to evidence token IDs.

    Args:
        mep: The evidence package.
        slot_to_id: Mapping from evidence slot name to token ID.
        num_slots: Total number of evidence slots.
        device: Target device for the tensor.

    Returns:
        LongTensor of shape [1, num_slots].
    """
    token_ids = torch.zeros(num_slots, dtype=torch.long)
    reasoning_dict = mep.reasoning.model_dump()

    # Map reasoning field NAMES to token IDs (0=padding, slot i -> i+1)
    for field_name in reasoning_dict:
        if field_name in slot_to_id:
            idx = slot_to_id[field_name]
            if idx < num_slots:
                token_ids[idx] = idx + 1

    result = token_ids.unsqueeze(0).to(dtype=torch.long)
    if device is not None:
        result = result.to(device)
    return result  # type: ignore[return-value]


def compute_score_cec(
    model: nn.Module,
    original_mep: MinimalEvidencePackage,
    weakened_meps: list[MinimalEvidencePackage],
    base_scores: dict[str, float],
    slot_to_id: dict[str, int],
    num_slots: int,
    z_v: torch.Tensor,
    base_logit: torch.Tensor,
) -> float:
    """Compute Score-CEC: fraction of cases where weakening changes the score.

    Score-CEC measures whether the residual readout responds to evidence changes.
    A perfect score means every weakening operation changes the fraud score.

    Args:
        model: GReaDReasoner model.
        original_mep: Original MEP.
        weakened_meps: List of weakened MEPs.
        base_scores: Unused (for API consistency).
        slot_to_id: Evidence slot name to token ID mapping.
        num_slots: Number of evidence slots.
        z_v: Node embedding tensor [1, H].
        base_logit: Base detector logit tensor [1].

    Returns:
        Score-CEC in [0, 1].
    """
    if not weakened_meps:
        return 0.0

    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        orig_ids = _mep_to_evidence_token_ids(original_mep, slot_to_id, num_slots, device=dev)
        orig_out = model(z_v, base_logit, orig_ids)
        orig_score = orig_out["final_logit"].item()

        changed = 0
        for w_mep in weakened_meps:
            w_ids = _mep_to_evidence_token_ids(w_mep, slot_to_id, num_slots, device=dev)
            w_out = model(z_v, base_logit, w_ids)
            w_score = w_out["final_logit"].item()
            if abs(w_score - orig_score) > 1e-6:
                changed += 1

    return float(changed / len(weakened_meps))


def compute_type_cec(
    model: nn.Module,
    original_mep: MinimalEvidencePackage,
    weakened_meps: list[MinimalEvidencePackage],
    slot_to_id: dict[str, int],
    num_slots: int,
    z_v: torch.Tensor,
    base_logit: torch.Tensor,
) -> float:
    """Compute Type-CEC: fraction of cases where weakening changes the risk type.

    Args:
        model: GReaDReasoner model.
        original_mep: Original MEP.
        weakened_meps: List of weakened MEPs.
        slot_to_id: Evidence slot name to token ID mapping.
        num_slots: Number of evidence slots.
        z_v: Node embedding tensor [1, H].
        base_logit: Base detector logit tensor [1].

    Returns:
        Type-CEC in [0, 1].
    """
    if not weakened_meps:
        return 0.0

    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        orig_ids = _mep_to_evidence_token_ids(original_mep, slot_to_id, num_slots, device=dev)
        orig_out = model(z_v, base_logit, orig_ids)
        orig_type = orig_out["type_logits"].argmax(dim=-1).item()

        changed = 0
        for w_mep in weakened_meps:
            w_ids = _mep_to_evidence_token_ids(w_mep, slot_to_id, num_slots, device=dev)
            w_out = model(z_v, base_logit, w_ids)
            w_type = w_out["type_logits"].argmax(dim=-1).item()
            if w_type != orig_type:
                changed += 1

    return float(changed / len(weakened_meps))


def compute_evidence_cec(
    model: nn.Module,
    original_mep: MinimalEvidencePackage,
    weakened_meps: list[MinimalEvidencePackage],
    slot_to_id: dict[str, int],
    num_slots: int,
    z_v: torch.Tensor,
    base_logit: torch.Tensor,
) -> float:
    """Compute Evidence-CEC: fraction of cases where weakening changes evidence masks.

    Evidence masks are binarized at 0.0 logit threshold.

    Args:
        model: GReaDReasoner model.
        original_mep: Original MEP.
        weakened_meps: List of weakened MEPs.
        slot_to_id: Evidence slot name to token ID mapping.
        num_slots: Number of evidence slots.
        z_v: Node embedding tensor [1, H].
        base_logit: Base detector logit tensor [1].

    Returns:
        Evidence-CEC in [0, 1].
    """
    if not weakened_meps:
        return 0.0

    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        orig_ids = _mep_to_evidence_token_ids(original_mep, slot_to_id, num_slots, device=dev)
        orig_out = model(z_v, base_logit, orig_ids)
        orig_pos = (orig_out["pos_mask_logits"] > 0).int()
        orig_neg = (orig_out["neg_mask_logits"] > 0).int()

        changed = 0
        for w_mep in weakened_meps:
            w_ids = _mep_to_evidence_token_ids(w_mep, slot_to_id, num_slots, device=dev)
            w_out = model(z_v, base_logit, w_ids)
            w_pos = (w_out["pos_mask_logits"] > 0).int()
            w_neg = (w_out["neg_mask_logits"] > 0).int()

            if not torch.equal(orig_pos, w_pos) or not torch.equal(orig_neg, w_neg):
                changed += 1

    return float(changed / len(weakened_meps))


def compute_tri_cec(
    model: nn.Module,
    meps: list[MinimalEvidencePackage],
    slot_to_id: dict[str, int],
    num_slots: int,
    z_v_batch: torch.Tensor,
    base_logit_batch: torch.Tensor,
    weaken_config: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compute tri-CEC over a batch of MEPs.

    Args:
        model: GReaDReasoner model.
        meps: List of MEPs to evaluate.
        slot_to_id: Evidence slot name to token ID mapping.
        num_slots: Number of evidence slots.
        z_v_batch: [N, H] node embeddings.
        base_logit_batch: [N] base detector logits.
        weaken_config: Custom weakening config. Defaults to DEFAULT_WEAKEN_CONFIG.

    Returns:
        Dict with "score_cec", "type_cec", "evidence_cec", and "n_samples".
    """
    if not meps:
        return {
            "score_cec": 0.0,
            "type_cec": 0.0,
            "evidence_cec": 0.0,
            "n_samples": 0,
        }

    score_cecs: list[float] = []
    type_cecs: list[float] = []
    evidence_cecs: list[float] = []

    for i, mep in enumerate(meps):
        # Generate all weakened variants for this MEP
        weakened_meps = _generate_weakened_variants(mep, weaken_config)
        if not weakened_meps:
            continue

        z_v = z_v_batch[i : i + 1]
        base_logit = base_logit_batch[i : i + 1]

        score_cecs.append(
            compute_score_cec(
                model, mep, weakened_meps, {}, slot_to_id, num_slots, z_v, base_logit
            )
        )
        type_cecs.append(
            compute_type_cec(
                model, mep, weakened_meps, slot_to_id, num_slots, z_v, base_logit
            )
        )
        evidence_cecs.append(
            compute_evidence_cec(
                model, mep, weakened_meps, slot_to_id, num_slots, z_v, base_logit
            )
        )

    n = len(score_cecs)
    return {
        "score_cec": sum(score_cecs) / n if n > 0 else 0.0,
        "type_cec": sum(type_cecs) / n if n > 0 else 0.0,
        "evidence_cec": sum(evidence_cecs) / n if n > 0 else 0.0,
        "n_samples": n,
    }


def _generate_weakened_variants(
    mep: MinimalEvidencePackage,
    weaken_config: dict[str, dict[str, str]] | None = None,
) -> list[MinimalEvidencePackage]:
    """Generate one weakened MEP per applicable field.

    Each variant weakens exactly one reasoning field, keeping others original.
    This allows per-field CEC measurement.

    Args:
        mep: Original MEP.
        weaken_config: Custom weakening config.

    Returns:
        List of weakened MEPs (one per applicable field).
    """
    if weaken_config is None:
        weaken_config = DEFAULT_WEAKEN_CONFIG

    reasoning_dict = mep.reasoning.model_dump()
    variants: list[MinimalEvidencePackage] = []

    for field, replacements in weaken_config.items():
        if field not in reasoning_dict:
            continue
        current_val = reasoning_dict[field]
        if current_val not in replacements:
            continue

        # Create a variant with only this field weakened
        variant_dict = dict(reasoning_dict)
        variant_dict[field] = replacements[current_val]

        variants.append(
            MinimalEvidencePackage(
                node_id=mep.node_id,
                detector_name=mep.detector_name,
                calibration=mep.calibration,  # NEVER modified
                reasoning=ReasoningChannel(**variant_dict),
            )
        )

    return variants

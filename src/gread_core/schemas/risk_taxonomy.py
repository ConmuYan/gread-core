from typing import Literal

RiskType = Literal[
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
]

RISK_TYPES: frozenset[str] = frozenset([
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
])

# Canonical ordered risk types — deterministic index mapping.
# All modules MUST use this list for index ↔ label conversion.
RISK_TYPES_ORDERED: list[str] = [
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
]

RISK_TYPE_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(RISK_TYPES_ORDERED)
}

EVIDENCE_SLOTS: frozenset[str] = frozenset([
    "uncertainty_level",
    "degree_level",
    "neighbor_consistency",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
    "counter_signal",
])

# Canonical ordered evidence slots — deterministic index mapping.
# All modules MUST use this list for slot ↔ index conversion.
EVIDENCE_SLOTS_ORDERED: list[str] = [
    "uncertainty_level",
    "degree_level",
    "neighbor_consistency",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
    "counter_signal",
]

EVIDENCE_SLOT_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(EVIDENCE_SLOTS_ORDERED)
}

FORBIDDEN_SUPPORT_IDS: frozenset[str] = frozenset([
    "prediction_score",
    "counter_signal",
])

FORBIDDEN_COUNTER_IDS: frozenset[str] = frozenset([
    "prediction_score",
])

SCORE_RELATED_IDS: frozenset[str] = frozenset([
    "prediction_score",
    "fraud_score",
    "probability_score",
    "base_score",
    "model_score",
])


def encode_evidence_slots(field_names: list[str], num_slots: int) -> list[int]:
    """Encode evidence field names to token IDs.

    Token 0 = padding, slot i -> token i + 1.
    """
    ids = [0] * num_slots
    for field_name in field_names:
        if field_name in EVIDENCE_SLOT_TO_INDEX:
            idx = EVIDENCE_SLOT_TO_INDEX[field_name]
            if idx < num_slots:
                ids[idx] = idx + 1
    return ids

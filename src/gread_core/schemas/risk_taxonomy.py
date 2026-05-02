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

EVIDENCE_SLOTS: frozenset[str] = frozenset([
    "uncertainty_level",
    "degree_level",
    "neighbor_consistency",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength",
    "counter_signal",
])

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

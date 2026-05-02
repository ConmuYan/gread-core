from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

UncertaintyLevel = Literal["low", "medium", "high"]
EvidenceStrength = Literal["weak", "moderate", "strong", "unavailable"]

_FORBIDDEN_SUPPORT_IDS: frozenset[str] = frozenset(["prediction_score", "counter_signal"])
_FORBIDDEN_COUNTER_IDS: frozenset[str] = frozenset(["prediction_score"])


class CalibrationChannel(BaseModel):
    prediction_score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)


class ReasoningChannel(BaseModel):
    uncertainty_level: UncertaintyLevel
    degree_level: str
    neighbor_consistency: str
    feature_neighbor_discrepancy: str
    detector_signal: str
    detector_signal_strength: EvidenceStrength
    counter_signal: str
    allowed_support_ids: list[str]
    allowed_counter_ids: list[str]

    @model_validator(mode="after")
    def _validate_role_rules(self) -> "ReasoningChannel":
        for sid in self.allowed_support_ids:
            if sid in _FORBIDDEN_SUPPORT_IDS:
                msg = f"Forbidden ID in allowed_support_ids: {sid}"
                raise ValueError(msg)
        for cid in self.allowed_counter_ids:
            if cid in _FORBIDDEN_COUNTER_IDS:
                msg = f"Forbidden ID in allowed_counter_ids: {cid}"
                raise ValueError(msg)
        if len(self.allowed_support_ids) != len(set(self.allowed_support_ids)):
            msg = "Duplicate IDs in allowed_support_ids"
            raise ValueError(msg)
        if len(self.allowed_counter_ids) != len(set(self.allowed_counter_ids)):
            msg = "Duplicate IDs in allowed_counter_ids"
            raise ValueError(msg)
        return self


class MinimalEvidencePackage(BaseModel):
    node_id: str
    detector_name: str
    calibration: CalibrationChannel
    reasoning: ReasoningChannel

    def to_teacher_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "detector_name": self.detector_name,
            "reasoning": self.reasoning.model_dump(),
        }

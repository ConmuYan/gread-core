from typing import Any, Literal

from pydantic import BaseModel, Field

RiskType = Literal[
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
]


class EvidenceRationaleRecord(BaseModel):
    risk_type: RiskType
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    summary: str

    def training_targets(self) -> dict[str, Any]:
        return {
            "risk_type": self.risk_type,
            "supporting_evidence": self.supporting_evidence,
            "counter_evidence": self.counter_evidence,
        }

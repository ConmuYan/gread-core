from typing import Any

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.schemas.risk_taxonomy import RISK_TYPES


def check_schema(
    err: EvidenceRationaleRecord,
    mep: MinimalEvidencePackage,
    config: dict[str, Any],
) -> tuple[bool, str]:
    if err.risk_type not in RISK_TYPES:
        return False, f"Unknown risk_type: {err.risk_type}"
    if not isinstance(err.supporting_evidence, list):
        return False, "supporting_evidence must be a list"
    if not isinstance(err.counter_evidence, list):
        return False, "counter_evidence must be a list"
    return True, ""

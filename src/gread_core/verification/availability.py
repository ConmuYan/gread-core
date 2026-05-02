from typing import Any

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage


def check_availability(
    err: EvidenceRationaleRecord,
    mep: MinimalEvidencePackage,
    config: dict[str, Any],
) -> tuple[bool, str]:
    available = (
        set(mep.reasoning.allowed_support_ids)
        | set(mep.reasoning.allowed_counter_ids)
    )
    for eid in err.supporting_evidence + err.counter_evidence:
        if eid not in available:
            return False, f"Evidence ID not available: {eid}"
    reasoning = mep.reasoning.model_dump()
    for eid in err.supporting_evidence:
        val = reasoning.get(eid)
        if val == "unavailable":
            return False, f"Supporting evidence has unavailable value: {eid}"
    return True, ""

from typing import Any

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.schemas.risk_taxonomy import SCORE_RELATED_IDS


def check_score_blindness(
    err: EvidenceRationaleRecord,
    mep: MinimalEvidencePackage,
    config: dict[str, Any],
) -> tuple[bool, str]:
    all_evidence = set(err.supporting_evidence) | set(err.counter_evidence)
    for score_id in SCORE_RELATED_IDS:
        if score_id in all_evidence:
            return False, f"Score-related ID in evidence: {score_id}"
    return True, ""

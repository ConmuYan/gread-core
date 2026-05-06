from typing import Any

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.schemas.risk_taxonomy import FORBIDDEN_COUNTER_IDS, FORBIDDEN_SUPPORT_IDS


def check_role_consistency(
    err: EvidenceRationaleRecord,
    mep: MinimalEvidencePackage,
    config: dict[str, Any],
) -> tuple[bool, str]:
    role_rules = config.get("role_rules", {})
    uncertainty_support_allowed = set(
        role_rules.get(
            "uncertainty_support_allowed_risk_types",
            ["weak_or_uncertain_evidence"],
        )
    )
    for eid in err.supporting_evidence:
        if eid not in mep.reasoning.allowed_support_ids:
            return False, f"supporting_evidence '{eid}' not in allowed_support_ids"
        if eid in FORBIDDEN_SUPPORT_IDS:
            return False, f"Forbidden as supporting_evidence: {eid}"
        if eid == "uncertainty_level" and err.risk_type not in uncertainty_support_allowed:
            return False, (
                "uncertainty_level can only support risk types: "
                f"{sorted(uncertainty_support_allowed)}"
            )
    for eid in err.counter_evidence:
        if eid not in mep.reasoning.allowed_counter_ids:
            return False, f"counter_evidence '{eid}' not in allowed_counter_ids"
        if eid in FORBIDDEN_COUNTER_IDS:
            return False, f"Forbidden as counter_evidence: {eid}"
    overlap = set(err.supporting_evidence) & set(err.counter_evidence)
    if overlap:
        return False, f"Evidence in both support and counter: {overlap}"
    return True, ""

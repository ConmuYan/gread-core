from typing import Any

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage


def check_contract(
    err: EvidenceRationaleRecord,
    mep: MinimalEvidencePackage,
    config: dict[str, Any],
) -> tuple[bool, str]:
    contracts = config.get("risk_types", {})
    risk_type = err.risk_type
    if risk_type not in contracts:
        return True, ""

    contract = contracts[risk_type]
    reasoning = mep.reasoning.model_dump()

    # required_any: at least one condition must be satisfied
    # A condition is satisfied ONLY IF:
    #   (a) the MEP field value matches the contract values, AND
    #   (b) the field is cited in err.supporting_evidence
    required_any = contract.get("required_any", [])
    if required_any:
        satisfied = False
        for cond in required_any:
            field: str = cond["field"]
            values: list[str] = cond["values"]
            if reasoning.get(field) in values and field in err.supporting_evidence:
                satisfied = True
                break
        if not satisfied:
            return False, f"No required condition satisfied for {risk_type}"

    # forbidden: MEP field values must not match
    forbidden = contract.get("forbidden", [])
    for cond in forbidden:
        field = cond["field"]
        values = cond["values"]
        if reasoning.get(field) in values:
            return False, f"Forbidden condition met for {risk_type}: {field}={reasoning.get(field)}"

    return True, ""

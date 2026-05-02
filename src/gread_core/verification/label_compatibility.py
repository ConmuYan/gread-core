from typing import Any

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage


def check_label_compatibility(
    err: EvidenceRationaleRecord,
    mep: MinimalEvidencePackage,
    config: dict[str, Any],
    label: int | None = None,
) -> tuple[bool, str]:
    lc = config.get("label_compatibility", {})
    if isinstance(lc, bool):
        lc = {"enabled": lc}
    if not lc.get("enabled", False) or label is None:
        return True, ""

    fraud_forbidden: list[str] = lc.get("fraud_forbidden_risk_types", [])
    benign_forbidden: list[str] = lc.get("benign_forbidden_risk_types", [])

    fraud_label: int = lc.get("fraud_label", 1)
    benign_label: int = lc.get("benign_label", 0)

    if label == fraud_label and err.risk_type in fraud_forbidden:
        return False, f"Risk type {err.risk_type} incompatible with fraud label"
    if label == benign_label and err.risk_type in benign_forbidden:
        return False, f"Risk type {err.risk_type} incompatible with benign label"
    return True, ""

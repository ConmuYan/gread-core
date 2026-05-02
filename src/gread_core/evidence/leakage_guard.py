from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.schemas.risk_taxonomy import SCORE_RELATED_IDS


def check_teacher_payload_score_blind(mep: MinimalEvidencePackage) -> tuple[bool, str]:
    payload = mep.to_teacher_payload()
    payload_str = str(payload)
    for score_id in SCORE_RELATED_IDS:
        if score_id in payload_str:
            return False, f"Score-related ID found in teacher payload: {score_id}"
    return True, ""


def check_mep_role_rules(mep: MinimalEvidencePackage) -> tuple[bool, str]:
    for sid in mep.reasoning.allowed_support_ids:
        if sid in ("prediction_score", "counter_signal"):
            return False, f"Forbidden ID in allowed_support_ids: {sid}"
    for cid in mep.reasoning.allowed_counter_ids:
        if cid == "prediction_score":
            return False, f"Forbidden ID in allowed_counter_ids: {cid}"
    return True, ""

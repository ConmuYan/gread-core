import json

from gread_core.evidence.leakage_guard import check_teacher_payload_score_blind
from gread_core.schemas.evidence import MinimalEvidencePackage


def test_teacher_payload_excludes_prediction_score(sample_mep: MinimalEvidencePackage) -> None:
    ok, reason = check_teacher_payload_score_blind(sample_mep)
    assert ok, reason


def test_teacher_payload_json_has_no_score(sample_mep: MinimalEvidencePackage) -> None:
    payload = sample_mep.to_teacher_payload()
    text = json.dumps(payload)
    assert "prediction_score" not in text
    assert "calibration" not in text
    assert "0.9" not in text

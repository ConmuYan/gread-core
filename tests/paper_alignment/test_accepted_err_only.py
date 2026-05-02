from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.verification.verifier import EvidenceContractVerifier


def test_rejected_err_has_no_training_contribution(
    verifier: EvidenceContractVerifier,
    sample_mep: MinimalEvidencePackage,
) -> None:
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["prediction_score"],
        counter_evidence=[],
        summary="Should be rejected for score leakage.",
    )
    result = verifier.verify(err, sample_mep, label=None)
    assert not result.accepted
    accepted_mask = 1 if result.accepted else 0
    assert accepted_mask == 0

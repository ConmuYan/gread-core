from gread_core.schemas.err import EvidenceRationaleRecord


def test_err_schema_separates_support_and_counter() -> None:
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal"],
        counter_evidence=["counter_signal"],
        summary="test",
    )
    targets = err.training_targets()
    assert "supporting_evidence" in targets
    assert "counter_evidence" in targets
    assert targets["supporting_evidence"] != targets["counter_evidence"]

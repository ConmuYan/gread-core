from gread_core.schemas.err import EvidenceRationaleRecord


def test_err_training_targets_excludes_summary() -> None:
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal"],
        counter_evidence=["counter_signal"],
        summary="This summary must not leak into training.",
    )
    targets = err.training_targets()
    assert "summary" not in targets
    assert set(targets.keys()) == {"risk_type", "supporting_evidence", "counter_evidence"}

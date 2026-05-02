import pytest
from pydantic import ValidationError

from gread_core.schemas.err import EvidenceRationaleRecord


@pytest.fixture
def sample_err() -> EvidenceRationaleRecord:
    return EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal", "neighbor_consistency"],
        counter_evidence=["counter_signal"],
        summary="Strong spectral evidence with low neighbor consistency.",
    )


def test_err_has_risk_type(sample_err: EvidenceRationaleRecord) -> None:
    assert sample_err.risk_type == "spectral_anomaly"


def test_err_has_supporting_and_counter(sample_err: EvidenceRationaleRecord) -> None:
    assert len(sample_err.supporting_evidence) == 2
    assert len(sample_err.counter_evidence) == 1


def test_err_summary_not_in_training_targets(sample_err: EvidenceRationaleRecord) -> None:
    targets = sample_err.training_targets()
    assert "summary" not in targets
    assert "risk_type" in targets
    assert "supporting_evidence" in targets
    assert "counter_evidence" in targets


def test_err_rejects_unknown_risk_type() -> None:
    with pytest.raises(ValidationError):
        EvidenceRationaleRecord(
            risk_type="unknown_type",
            supporting_evidence=[],
            counter_evidence=[],
            summary="test",
        )


def test_err_rejects_missing_risk_type() -> None:
    with pytest.raises(ValidationError):
        EvidenceRationaleRecord(
            supporting_evidence=[],
            counter_evidence=[],
            summary="test",
        )  # type: ignore[call-arg]


def test_err_default_evidence_lists() -> None:
    err = EvidenceRationaleRecord(
        risk_type="weak_or_uncertain_evidence",
        summary="No strong evidence.",
    )
    assert err.supporting_evidence == []
    assert err.counter_evidence == []

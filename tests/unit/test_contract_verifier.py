from pathlib import Path
from typing import Any

import pytest
import yaml

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)
from gread_core.verification.verifier import EvidenceContractVerifier


@pytest.fixture
def contract_config() -> dict[str, Any]:
    config_path = Path("configs/contracts/gread_v1.yaml")
    return yaml.safe_load(config_path.read_text())


@pytest.fixture
def verifier(contract_config: dict[str, Any]) -> EvidenceContractVerifier:
    return EvidenceContractVerifier(contract_config)


def _make_mep(**overrides: Any) -> MinimalEvidencePackage:
    reasoning: dict[str, Any] = {
        "uncertainty_level": "low",
        "degree_level": "high",
        "neighbor_consistency": "low",
        "feature_neighbor_discrepancy": "high",
        "detector_signal": "high_frequency_response_high",
        "detector_signal_strength": "strong",
        "counter_signal": "benign_neighbor_signal_low",
        "allowed_support_ids": [
            "degree_level",
            "neighbor_consistency",
            "feature_neighbor_discrepancy",
            "detector_signal",
            "detector_signal_strength",
        ],
        "allowed_counter_ids": ["counter_signal", "uncertainty_level"],
    }
    for k, v in overrides.items():
        reasoning[k] = v
    return MinimalEvidencePackage(
        node_id="n1",
        detector_name="bwgnn",
        calibration=CalibrationChannel(prediction_score=0.9, uncertainty=0.1),
        reasoning=ReasoningChannel(**reasoning),
    )


# --- Core acceptance/rejection ---


def test_valid_spectral_anomaly_accepted(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep()
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal", "neighbor_consistency"],
        counter_evidence=["counter_signal"],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert result.accepted, f"Rejected: {result.reasons}"


def test_counter_signal_as_supporting_rejected(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep()
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["counter_signal"],
        counter_evidence=[],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted


def test_prediction_score_in_evidence_rejected(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep()
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["prediction_score"],
        counter_evidence=[],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted


def test_unavailable_detector_signal_spectral_rejected(
    verifier: EvidenceContractVerifier,
) -> None:
    mep = _make_mep(
        detector_signal="unavailable",
        detector_signal_strength="unavailable",
    )
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["neighbor_consistency"],
        counter_evidence=[],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted


def test_unknown_evidence_id_rejected(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep(allowed_support_ids=["nonexistent_evidence", "degree_level"])
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["nonexistent_evidence"],
        counter_evidence=[],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted


def test_benign_label_rejects_strong_risk(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep()
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal", "neighbor_consistency"],
        counter_evidence=["counter_signal"],
        summary="test",
    )
    result = verifier.verify(err, mep, label=0)
    assert not result.accepted


def test_fraud_label_rejects_weak_risk(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep(
        uncertainty_level="low",
        allowed_support_ids=["degree_level"],
    )
    err = EvidenceRationaleRecord(
        risk_type="weak_or_uncertain_evidence",
        supporting_evidence=[],
        counter_evidence=[],
        summary="test",
    )
    result = verifier.verify(err, mep, label=1)
    assert not result.accepted


def test_summary_changes_do_not_affect_verification(
    verifier: EvidenceContractVerifier,
) -> None:
    mep = _make_mep()
    err1 = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal"],
        counter_evidence=["counter_signal"],
        summary="Summary A",
    )
    err2 = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal"],
        counter_evidence=["counter_signal"],
        summary="Summary B is completely different",
    )
    r1 = verifier.verify(err1, mep, label=None)
    r2 = verifier.verify(err2, mep, label=None)
    assert r1.accepted == r2.accepted


def test_verifier_is_deterministic(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep()
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal"],
        counter_evidence=["counter_signal"],
        summary="test",
    )
    r1 = verifier.verify(err, mep, label=None)
    r2 = verifier.verify(err, mep, label=None)
    assert r1.accepted == r2.accepted
    assert r1.reasons == r2.reasons


def test_supporting_counter_overlap_rejected(verifier: EvidenceContractVerifier) -> None:
    mep = _make_mep(
        allowed_support_ids=["detector_signal"],
        allowed_counter_ids=["detector_signal", "counter_signal"],
    )
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal"],
        counter_evidence=["detector_signal"],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted


# --- Strengthened contract tests ---


def test_spectral_anomaly_rejected_when_detector_signal_not_cited(
    verifier: EvidenceContractVerifier,
) -> None:
    """detector_signal is present in MEP but NOT cited as supporting evidence."""
    mep = _make_mep(detector_signal="high_frequency_response_high")
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["neighbor_consistency"],
        counter_evidence=["counter_signal"],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted


def test_uncertainty_alone_cannot_support_strong_fraud_risk(
    verifier: EvidenceContractVerifier,
) -> None:
    mep = _make_mep(
        uncertainty_level="high",
        detector_signal="unavailable",
        detector_signal_strength="unavailable",
        allowed_support_ids=["uncertainty_level", "degree_level"],
    )
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["uncertainty_level"],
        counter_evidence=[],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted


def test_supporting_evidence_with_unavailable_value_rejected(
    verifier: EvidenceContractVerifier,
) -> None:
    mep = _make_mep(detector_signal_strength="unavailable")
    err = EvidenceRationaleRecord(
        risk_type="spectral_anomaly",
        supporting_evidence=["detector_signal_strength"],
        counter_evidence=[],
        summary="test",
    )
    result = verifier.verify(err, mep, label=None)
    assert not result.accepted

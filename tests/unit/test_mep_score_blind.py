import json

import pytest
from pydantic import ValidationError

from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)


@pytest.fixture
def valid_reasoning_kwargs() -> dict[str, object]:
    return {
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


@pytest.fixture
def sample_mep(valid_reasoning_kwargs: dict[str, object]) -> MinimalEvidencePackage:
    return MinimalEvidencePackage(
        node_id="test_node_001",
        detector_name="bwgnn",
        calibration=CalibrationChannel(prediction_score=0.83, uncertainty=0.17),
        reasoning=ReasoningChannel(**valid_reasoning_kwargs),  # type: ignore[arg-type]
    )


# --- Core score-blind tests ---


def test_prediction_score_exists_in_calibration(sample_mep: MinimalEvidencePackage) -> None:
    assert sample_mep.calibration.prediction_score == 0.83


def test_prediction_score_absent_from_teacher_payload(
    sample_mep: MinimalEvidencePackage,
) -> None:
    payload = sample_mep.to_teacher_payload()
    payload_str = json.dumps(payload)
    assert "prediction_score" not in payload_str
    assert "calibration" not in payload_str
    assert "0.83" not in payload_str


def test_counter_signal_not_in_allowed_support_ids(
    sample_mep: MinimalEvidencePackage,
) -> None:
    assert "counter_signal" not in sample_mep.reasoning.allowed_support_ids


def test_teacher_payload_serializable_json(sample_mep: MinimalEvidencePackage) -> None:
    payload = sample_mep.to_teacher_payload()
    serialized = json.dumps(payload)
    assert "test_node_001" in serialized
    assert "bwgnn" in serialized


def test_teacher_payload_contains_only_reasoning(sample_mep: MinimalEvidencePackage) -> None:
    payload = sample_mep.to_teacher_payload()
    assert "reasoning" in payload
    assert "node_id" in payload
    assert "detector_name" in payload
    assert "calibration" not in payload
    assert "prediction_score" not in json.dumps(payload)


def test_calibration_channel_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CalibrationChannel(prediction_score=1.5, uncertainty=0.1)
    with pytest.raises(ValidationError):
        CalibrationChannel(prediction_score=-0.1, uncertainty=0.1)


def test_reasoning_channel_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        ReasoningChannel(uncertainty_level="low")  # type: ignore[call-arg]


# --- Pydantic validator tests (construction-time guards) ---


def test_rejects_prediction_score_in_support_ids(
    valid_reasoning_kwargs: dict[str, object],
) -> None:
    kwargs = {**valid_reasoning_kwargs, "allowed_support_ids": ["prediction_score"]}
    with pytest.raises(ValidationError, match="prediction_score"):
        ReasoningChannel(**kwargs)  # type: ignore[arg-type]


def test_rejects_counter_signal_in_support_ids(
    valid_reasoning_kwargs: dict[str, object],
) -> None:
    kwargs = {**valid_reasoning_kwargs, "allowed_support_ids": ["counter_signal"]}
    with pytest.raises(ValidationError, match="counter_signal"):
        ReasoningChannel(**kwargs)  # type: ignore[arg-type]


def test_rejects_prediction_score_in_counter_ids(
    valid_reasoning_kwargs: dict[str, object],
) -> None:
    kwargs = {**valid_reasoning_kwargs, "allowed_counter_ids": ["prediction_score"]}
    with pytest.raises(ValidationError, match="prediction_score"):
        ReasoningChannel(**kwargs)  # type: ignore[arg-type]


def test_rejects_duplicate_support_ids(
    valid_reasoning_kwargs: dict[str, object],
) -> None:
    kwargs = {
        **valid_reasoning_kwargs,
        "allowed_support_ids": ["degree_level", "degree_level"],
    }
    with pytest.raises(ValidationError, match=r"[Dd]uplicate"):
        ReasoningChannel(**kwargs)  # type: ignore[arg-type]

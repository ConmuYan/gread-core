"""PromptBuilder unit tests: score-blind checks."""

from __future__ import annotations

import pytest

from gread_core.llm.prompt_builder import PromptBuilder
from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def sample_mep() -> MinimalEvidencePackage:
    return MinimalEvidencePackage(
        node_id="n1",
        detector_name="bwgnn",
        calibration=CalibrationChannel(prediction_score=0.95, uncertainty=0.05),
        reasoning=ReasoningChannel(
            uncertainty_level="low",
            degree_level="high",
            neighbor_consistency="low",
            feature_neighbor_discrepancy="high",
            detector_signal="high_frequency_response_high",
            detector_signal_strength="strong",
            counter_signal="benign_neighbor_signal_low",
            allowed_support_ids=[
                "degree_level",
                "neighbor_consistency",
                "detector_signal",
            ],
            allowed_counter_ids=["counter_signal"],
        ),
    )


def test_prediction_score_not_in_prompt(
    builder: PromptBuilder, sample_mep: MinimalEvidencePackage
) -> None:
    payload = sample_mep.to_teacher_payload()
    prompt = builder.build(payload)
    assert "prediction_score" not in prompt, (
        "prediction_score must not appear in the LLM prompt"
    )
    assert "0.95" not in prompt, (
        "prediction_score value must not leak into prompt"
    )


def test_uncertainty_not_in_prompt(
    builder: PromptBuilder, sample_mep: MinimalEvidencePackage
) -> None:
    payload = sample_mep.to_teacher_payload()
    prompt = builder.build(payload)
    # calibration.uncertainty is also excluded via to_teacher_payload
    assert "0.05" not in prompt


def test_prompt_contains_reasoning_fields(
    builder: PromptBuilder, sample_mep: MinimalEvidencePackage
) -> None:
    payload = sample_mep.to_teacher_payload()
    prompt = builder.build(payload)
    assert "high_frequency_response_high" in prompt
    assert "benign_neighbor_signal_low" in prompt
    assert "bwgnn" in prompt


def test_prompt_contains_json_schema(
    builder: PromptBuilder, sample_mep: MinimalEvidencePackage
) -> None:
    payload = sample_mep.to_teacher_payload()
    prompt = builder.build(payload)
    assert '"risk_type"' in prompt
    assert '"supporting_evidence"' in prompt
    assert '"counter_evidence"' in prompt
    assert '"summary"' in prompt


def test_prompt_contains_taxonomy(
    builder: PromptBuilder, sample_mep: MinimalEvidencePackage
) -> None:
    payload = sample_mep.to_teacher_payload()
    prompt = builder.build(payload)
    for risk in [
        "structural_discrepancy",
        "camouflage_neighbor",
        "spectral_anomaly",
        "feature_structure_conflict",
        "relation_or_burst_anomaly",
        "weak_or_uncertain_evidence",
    ]:
        assert risk in prompt


def test_prompt_contains_node_id(
    builder: PromptBuilder, sample_mep: MinimalEvidencePackage
) -> None:
    payload = sample_mep.to_teacher_payload()
    prompt = builder.build(payload)
    assert "n1" in prompt

"""Integration tests for the tri-CEC pipeline."""

from __future__ import annotations

import pytest
import torch

from gread_core.evaluation.cec import (
    _generate_weakened_variants,
    compute_evidence_cec,
    compute_score_cec,
    compute_tri_cec,
    compute_type_cec,
    weaken_evidence,
)
from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.reasoner import GReaDReasoner
from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)


@pytest.fixture
def sample_mep() -> MinimalEvidencePackage:
    return MinimalEvidencePackage(
        node_id="n1",
        detector_name="bwgnn",
        calibration=CalibrationChannel(prediction_score=0.9, uncertainty=0.1),
        reasoning=ReasoningChannel(
            uncertainty_level="low",
            degree_level="burst",
            neighbor_consistency="low",
            feature_neighbor_discrepancy="high",
            detector_signal="high_frequency_response_high",
            detector_signal_strength="strong",
            counter_signal="benign_neighbor_signal_low",
            allowed_support_ids=[
                "degree_level",
                "neighbor_consistency",
                "feature_neighbor_discrepancy",
                "detector_signal",
                "detector_signal_strength",
            ],
            allowed_counter_ids=["counter_signal", "uncertainty_level"],
        ),
    )


@pytest.fixture
def reasoner() -> GReaDReasoner:
    encoder = EvidenceEncoder(vocab_size=42, embed_dim=16, num_slots=32, output_dim=32)
    return GReaDReasoner(
        hidden_dim=16,
        evidence_encoder=encoder,
        num_risk_types=6,
        num_evidence_slots=32,
        rho=0.1,
    )


class TestWeakenEvidence:
    def test_calibration_preserved(self, sample_mep: MinimalEvidencePackage) -> None:
        """CRITICAL: weakening must never touch calibration channel."""
        weakened = weaken_evidence(sample_mep)
        assert weakened.calibration.prediction_score == sample_mep.calibration.prediction_score
        assert weakened.calibration.uncertainty == sample_mep.calibration.uncertainty

    def test_detector_signal_weakened(self, sample_mep: MinimalEvidencePackage) -> None:
        weakened = weaken_evidence(sample_mep)
        assert weakened.reasoning.detector_signal != "high_frequency_response_high"

    def test_strength_weakened(self, sample_mep: MinimalEvidencePackage) -> None:
        weakened = weaken_evidence(sample_mep)
        assert weakened.reasoning.detector_signal_strength == "weak"

    def test_node_id_preserved(self, sample_mep: MinimalEvidencePackage) -> None:
        weakened = weaken_evidence(sample_mep)
        assert weakened.node_id == sample_mep.node_id

    def test_custom_config(self, sample_mep: MinimalEvidencePackage) -> None:
        config = {"detector_signal_strength": {"strong": "moderate"}}
        weakened = weaken_evidence(sample_mep, weaken_config=config)
        assert weakened.reasoning.detector_signal_strength == "moderate"
        # Other fields unchanged
        assert weakened.reasoning.detector_signal == sample_mep.reasoning.detector_signal


class TestGenerateWeakenedVariants:
    def test_generates_per_field(self, sample_mep: MinimalEvidencePackage) -> None:
        variants = _generate_weakened_variants(sample_mep)
        # sample_mep has: detector_signal, detector_signal_strength,
        # neighbor_consistency, feature_neighbor_discrepancy, degree_level
        # all of which have matching values in DEFAULT_WEAKEN_CONFIG
        assert len(variants) >= 3

    def test_each_variant_differs_in_one_field(
        self, sample_mep: MinimalEvidencePackage
    ) -> None:
        variants = _generate_weakened_variants(sample_mep)
        original = sample_mep.reasoning.model_dump()
        for v in variants:
            v_dict = v.reasoning.model_dump()
            diffs = sum(1 for k in original if original[k] != v_dict[k])
            assert diffs == 1, "Each variant should differ in exactly one field"


class TestScoreCEC:
    def test_runs_without_error(
        self, reasoner: GReaDReasoner, sample_mep: MinimalEvidencePackage
    ) -> None:
        z_v = torch.randn(1, 16)
        base_logit = torch.randn(1)
        slot_to_id = {f"slot_{i}": i for i in range(32)}
        weakened = _generate_weakened_variants(sample_mep)
        result = compute_score_cec(
            reasoner, sample_mep, weakened, {}, slot_to_id, 32, z_v, base_logit
        )
        assert 0.0 <= result <= 1.0

    def test_empty_weakened_returns_zero(
        self, reasoner: GReaDReasoner, sample_mep: MinimalEvidencePackage
    ) -> None:
        z_v = torch.randn(1, 16)
        base_logit = torch.randn(1)
        slot_to_id = {f"slot_{i}": i for i in range(32)}
        result = compute_score_cec(
            reasoner, sample_mep, [], {}, slot_to_id, 32, z_v, base_logit
        )
        assert result == 0.0


class TestTypeCEC:
    def test_runs_without_error(
        self, reasoner: GReaDReasoner, sample_mep: MinimalEvidencePackage
    ) -> None:
        z_v = torch.randn(1, 16)
        base_logit = torch.randn(1)
        slot_to_id = {f"slot_{i}": i for i in range(32)}
        weakened = _generate_weakened_variants(sample_mep)
        result = compute_type_cec(
            reasoner, sample_mep, weakened, slot_to_id, 32, z_v, base_logit
        )
        assert 0.0 <= result <= 1.0


class TestEvidenceCEC:
    def test_runs_without_error(
        self, reasoner: GReaDReasoner, sample_mep: MinimalEvidencePackage
    ) -> None:
        z_v = torch.randn(1, 16)
        base_logit = torch.randn(1)
        slot_to_id = {f"slot_{i}": i for i in range(32)}
        weakened = _generate_weakened_variants(sample_mep)
        result = compute_evidence_cec(
            reasoner, sample_mep, weakened, slot_to_id, 32, z_v, base_logit
        )
        assert 0.0 <= result <= 1.0


class TestTriCEC:
    def test_full_pipeline(
        self, reasoner: GReaDReasoner, sample_mep: MinimalEvidencePackage
    ) -> None:
        meps = [sample_mep] * 3
        z_v = torch.randn(3, 16)
        base_logit = torch.randn(3)
        slot_to_id = {f"slot_{i}": i for i in range(32)}
        result = compute_tri_cec(reasoner, meps, slot_to_id, 32, z_v, base_logit)

        assert "score_cec" in result
        assert "type_cec" in result
        assert "evidence_cec" in result
        assert result["n_samples"] == 3
        assert 0.0 <= result["score_cec"] <= 1.0
        assert 0.0 <= result["type_cec"] <= 1.0
        assert 0.0 <= result["evidence_cec"] <= 1.0

    def test_empty_meps(self, reasoner: GReaDReasoner) -> None:
        slot_to_id = {f"slot_{i}": i for i in range(32)}
        result = compute_tri_cec(
            reasoner, [], slot_to_id, 32, torch.empty(0, 16), torch.empty(0)
        )
        assert result["n_samples"] == 0
        assert result["score_cec"] == 0.0

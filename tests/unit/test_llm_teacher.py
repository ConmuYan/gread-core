from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from gread_core.llm.teacher import LLMTeacher
from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)
from gread_core.verification.verifier import EvidenceContractVerifier


class _BatchClient:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def complete(self, prompt: str) -> str:
        raise AssertionError("complete() should not be used in batched teacher path")

    def complete_batch(self, prompts: list[str]) -> list[str]:
        self.batch_sizes.append(len(prompts))
        return [
            json.dumps(
                {
                    "risk_type": "spectral_anomaly",
                    "supporting_evidence": ["detector_signal", "detector_signal_strength"],
                    "counter_evidence": ["counter_signal"],
                    "summary": "High spectral response with low neighbor consistency.",
                }
            )
            for _ in prompts
        ]


class _StaticBatchClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, prompt: str) -> str:
        raise AssertionError("complete() should not be used in batched teacher path")

    def complete_batch(self, prompts: list[str]) -> list[str]:
        return self.responses[: len(prompts)]


def _make_mep(node_id: str) -> MinimalEvidencePackage:
    return MinimalEvidencePackage(
        node_id=node_id,
        detector_name="bwgnn",
        calibration=CalibrationChannel(prediction_score=0.9, uncertainty=0.1),
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
                "feature_neighbor_discrepancy",
                "detector_signal",
                "detector_signal_strength",
            ],
            allowed_counter_ids=["counter_signal", "uncertainty_level"],
        ),
    )


def test_teacher_uses_complete_batch(tmp_path: Path) -> None:
    contract_config: dict[str, Any] = yaml.safe_load(
        Path("configs/contracts/gread_v1.yaml").read_text()
    )
    verifier = EvidenceContractVerifier(contract_config)
    client = _BatchClient()
    teacher = LLMTeacher(
        client=client,
        verifier=verifier,
        cache_dir=str(tmp_path / "err_cache"),
        batch_size=2,
        show_progress=False,
    )

    meps = [_make_mep("n1"), _make_mep("n2"), _make_mep("n3")]
    results = teacher.generate_err(meps, labels=[1, 1, 1])

    assert len(results) == 3
    assert client.batch_sizes == [2, 1]


def test_teacher_attempts_preserve_rejected_raw_and_reasons(tmp_path: Path) -> None:
    contract_config: dict[str, Any] = yaml.safe_load(
        Path("configs/contracts/gread_v1.yaml").read_text()
    )
    verifier = EvidenceContractVerifier(contract_config)
    accepted_raw = json.dumps(
        {
            "risk_type": "spectral_anomaly",
            "supporting_evidence": ["detector_signal", "detector_signal_strength"],
            "counter_evidence": ["counter_signal"],
            "summary": "High spectral response with low neighbor consistency.",
        }
    )
    rejected_raw = json.dumps(
        {
            "risk_type": "spectral_anomaly",
            "supporting_evidence": ["prediction_score"],
            "counter_evidence": [],
            "summary": "Score-only explanation.",
        }
    )
    parse_failed_raw = "not json"
    teacher = LLMTeacher(
        client=_StaticBatchClient([accepted_raw, rejected_raw, parse_failed_raw]),
        verifier=verifier,
        cache_dir=str(tmp_path / "err_cache"),
        batch_size=3,
        show_progress=False,
    )

    attempts = teacher.generate_err_attempts(
        [_make_mep("n1"), _make_mep("n2"), _make_mep("n3")],
        labels=[1, 1, 1],
    )

    assert [attempt.node_id for attempt in attempts] == ["n1", "n2", "n3"]
    assert attempts[0].verification.accepted
    assert attempts[1].raw_response == rejected_raw
    assert attempts[1].err is not None
    assert not attempts[1].verification.accepted
    assert "Score-related ID in evidence: prediction_score" in attempts[1].verification.reasons
    assert attempts[2].raw_response == parse_failed_raw
    assert attempts[2].err is None
    assert not attempts[2].verification.accepted
    assert attempts[2].parse_error is not None

"""Integration test: LLMTeacher replay from cache without network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gread_core.llm.clients import ReplayClient
from gread_core.llm.prompt_builder import PromptBuilder
from gread_core.llm.teacher import LLMTeacher
from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)
from gread_core.verification.verifier import EvidenceContractVerifier


def _make_mep(node_id: str = "n1") -> MinimalEvidencePackage:
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


def _valid_err_json() -> str:
    return json.dumps(
        {
            "risk_type": "spectral_anomaly",
            "supporting_evidence": ["detector_signal", "detector_signal_strength"],
            "counter_evidence": ["counter_signal"],
            "summary": "High spectral response with low neighbor consistency.",
        }
    )


def _seed_cache(cache_dir: Path, mep: MinimalEvidencePackage) -> None:
    """Pre-populate cache so ReplayClient can find the response."""
    builder = PromptBuilder()
    prompt = builder.build(mep.to_teacher_payload())
    cache_file = cache_dir / "cache.jsonl"
    import hashlib

    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    entry = {"prompt_hash": key, "response": _valid_err_json()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def test_replay_generates_accepted_err(tmp_path: Path) -> None:
    cache_dir = tmp_path / "err_cache"
    mep = _make_mep()
    _seed_cache(cache_dir, mep)

    contract_config: dict[str, Any] = yaml.safe_load(
        Path("configs/contracts/gread_v1.yaml").read_text()
    )
    verifier = EvidenceContractVerifier(contract_config)
    client = ReplayClient(cache_dir)
    teacher = LLMTeacher(client=client, verifier=verifier, cache_dir=str(cache_dir))

    results = teacher.generate_err([mep])
    assert len(results) == 1
    assert results[0].verification.accepted
    assert results[0].err.risk_type == "spectral_anomaly"


def test_replay_no_duplicate_call(tmp_path: Path) -> None:
    """Same prompt hash should not trigger a second client call."""
    cache_dir = tmp_path / "err_cache"
    mep = _make_mep()
    _seed_cache(cache_dir, mep)

    contract_config: dict[str, Any] = yaml.safe_load(
        Path("configs/contracts/gread_v1.yaml").read_text()
    )
    verifier = EvidenceContractVerifier(contract_config)
    client = ReplayClient(cache_dir)
    teacher = LLMTeacher(client=client, verifier=verifier, cache_dir=str(cache_dir))

    # Run twice with the same MEP — both should hit cache
    results1 = teacher.generate_err([mep])
    results2 = teacher.generate_err([mep])
    assert len(results1) == 1
    assert len(results2) == 1


def test_replay_without_network_access(tmp_path: Path) -> None:
    """ReplayClient reads from disk only, no network dependency."""
    cache_dir = tmp_path / "err_cache"
    mep = _make_mep()
    _seed_cache(cache_dir, mep)

    client = ReplayClient(cache_dir)
    builder = PromptBuilder()
    prompt = builder.build(mep.to_teacher_payload())
    result = client.complete(prompt)
    parsed = json.loads(result)
    assert parsed["risk_type"] == "spectral_anomaly"


def test_replay_cache_miss_raises(tmp_path: Path) -> None:
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir()
    client = ReplayClient(cache_dir)
    with pytest.raises(KeyError, match="No cached response"):
        client.complete("unseen prompt")


def test_score_blind_in_replay_prompt(tmp_path: Path) -> None:
    """The prompt stored in cache must never contain prediction_score."""
    cache_dir = tmp_path / "err_cache"
    mep = _make_mep()
    _seed_cache(cache_dir, mep)

    builder = PromptBuilder()
    prompt = builder.build(mep.to_teacher_payload())
    assert "prediction_score" not in prompt
    assert "0.9" not in prompt

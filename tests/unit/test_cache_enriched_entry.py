"""Unit test for enriched cache entries with verification metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gread_core.llm.cache import PromptCache


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def test_enriched_entry_contains_required_fields(cache_dir: Path) -> None:
    """After put() with metadata, the JSONL entry must contain
    prompt_hash, response, verification_result, and contract_version."""
    cache = PromptCache(cache_dir)
    prompt = "test prompt for enrichment"
    response = '{"node_id": "n1", "reasoning_chain": []}'

    cache.put(
        prompt,
        response,
        verification_result="accepted",
        contract_version="gread_v1",
    )

    # Read back the JSONL file and parse the last entry
    lines = (cache_dir / "cache.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])

    # prompt_hash: str, 16 chars (sha256 hex prefix)
    assert "prompt_hash" in entry
    assert isinstance(entry["prompt_hash"], str)
    assert len(entry["prompt_hash"]) == 16

    # response: str
    assert "response" in entry
    assert isinstance(entry["response"], str)
    assert entry["response"] == response

    # verification_result == "accepted"
    assert "verification_result" in entry
    assert entry["verification_result"] == "accepted"

    # contract_version == "gread_v1"
    assert "contract_version" in entry
    assert entry["contract_version"] == "gread_v1"


def test_enriched_entry_rejected(cache_dir: Path) -> None:
    """A rejected verification result is also persisted."""
    cache = PromptCache(cache_dir)
    cache.put(
        "another prompt",
        '{"reasoning_chain": []}',
        verification_result="rejected",
        contract_version="gread_v1",
    )

    lines = (cache_dir / "cache.jsonl").read_text().strip().splitlines()
    entry = json.loads(lines[0])
    assert entry["verification_result"] == "rejected"
    assert entry["contract_version"] == "gread_v1"


def test_bare_put_has_no_metadata(cache_dir: Path) -> None:
    """A plain put() without metadata should not include extra fields."""
    cache = PromptCache(cache_dir)
    cache.put("bare prompt", "bare response")

    lines = (cache_dir / "cache.jsonl").read_text().strip().splitlines()
    entry = json.loads(lines[0])

    assert "prompt_hash" in entry
    assert "response" in entry
    assert "verification_result" not in entry
    assert "contract_version" not in entry

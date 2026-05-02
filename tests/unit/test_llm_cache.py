"""Unit tests for PromptCache: hash-keyed, no duplicates, replay mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gread_core.llm.cache import PromptCache


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def test_put_and_get(cache_dir: Path) -> None:
    cache = PromptCache(cache_dir)
    cache.put("hello prompt", "hello response")
    assert cache.get("hello prompt") == "hello response"


def test_cache_miss_returns_none(cache_dir: Path) -> None:
    cache = PromptCache(cache_dir)
    assert cache.get("nonexistent") is None


def test_duplicate_put_does_not_duplicate_file(cache_dir: Path) -> None:
    cache = PromptCache(cache_dir)
    cache.put("dup prompt", "response a")
    cache.put("dup prompt", "response b")
    # Second put for same key is a no-op
    assert cache.get("dup prompt") == "response a"
    lines = (cache_dir / "cache.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_persistence_across_instances(cache_dir: Path) -> None:
    PromptCache(cache_dir).put("persist", "data")
    cache2 = PromptCache(cache_dir)
    assert cache2.get("persist") == "data"


def test_multiple_entries(cache_dir: Path) -> None:
    cache = PromptCache(cache_dir)
    cache.put("prompt1", "response1")
    cache.put("prompt2", "response2")
    assert cache.get("prompt1") == "response1"
    assert cache.get("prompt2") == "response2"


def test_cache_key_is_hash_not_node_id(cache_dir: Path) -> None:
    """Cache key must be prompt hash, not node_id."""
    cache = PromptCache(cache_dir)
    prompt_text = "some prompt text"
    cache.put(prompt_text, "ok")
    raw = (cache_dir / "cache.jsonl").read_text()
    entry = json.loads(raw.strip())
    assert "prompt_hash" in entry
    assert entry["prompt_hash"] != "n1"
    assert len(entry["prompt_hash"]) == 16

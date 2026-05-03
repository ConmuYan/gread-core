"""Prompt-hash-keyed cache for LLM completions.

Supports two modes:
- **write** (default): store completions alongside existing entries.
- **replay**: return cached completions, raise on cache miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "cache.jsonl"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class PromptCache:
    """Append-only JSONL cache keyed by prompt hash."""

    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _CACHE_FILENAME
        self._index: dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get(self, prompt: str) -> str | None:
        """Return cached response or ``None``."""
        return self._index.get(_prompt_hash(prompt))

    def put(
        self,
        prompt: str,
        response: str,
        *,
        payload_hash: str | None = None,
        model: str | None = None,
        verification_result: str | None = None,
        contract_version: str | None = None,
    ) -> None:
        """Append a new entry and update the in-memory index.

        If the key already exists *and* new metadata is provided, appends an
        enriched entry so the JSONL captures verification context.
        """
        key = _prompt_hash(prompt)
        has_metadata = any(
            v is not None
            for v in (payload_hash, model, verification_result, contract_version)
        )
        if key in self._index and not has_metadata:
            return
        entry: dict[str, str] = {"prompt_hash": key, "response": response}
        if payload_hash is not None:
            entry["payload_hash"] = payload_hash
        if model is not None:
            entry["model"] = model
        if verification_result is not None:
            entry["verification_result"] = verification_result
        if contract_version is not None:
            entry["contract_version"] = contract_version
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._index[key] = response
        logger.debug("Cached response for prompt_hash=%s", key)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._index[entry["prompt_hash"]] = entry["response"]

"""LLM client abstraction: protocol, OpenAI adapter, and replay client."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    def complete(self, prompt: str) -> str:
        """Return raw string completion for *prompt*."""
        ...


class OpenAIClient:
    """Thin wrapper around the OpenAI ChatCompletion API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            msg = (
                "openai package is required for OpenAIClient. "
                "Install with: pip install openai"
            )
            raise ImportError(msg) from exc

        self._client = OpenAI()
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries

    def complete(self, prompt: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1 + self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                logger.warning("OpenAI call attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(
            f"OpenAI completion failed after {1 + self._max_retries} attempts"
        ) from last_exc


class StubClient:
    """Deterministic stub that returns a valid ERR JSON for any prompt.

    Used for offline smoke testing when no LLM cache is available.
    """

    _STUB_RESPONSE = json.dumps({
        "risk_type": "structural_discrepancy",
        "supporting_evidence": ["degree_level", "neighbor_consistency"],
        "counter_evidence": ["counter_signal"],
        "summary": "Stub ERR for offline smoke testing.",
    })

    def complete(self, prompt: str) -> str:
        return self._STUB_RESPONSE


class ReplayClient:
    """Replay cached completions keyed by prompt hash.

    Used during integration tests and offline cache replay to avoid
    any network calls.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        from gread_core.llm.cache import PromptCache

        self._cache = PromptCache(cache_dir)

    def complete(self, prompt: str) -> str:
        cached = self._cache.get(prompt)
        if cached is not None:
            return cached
        msg = (
            "No cached response for prompt hash. "
            "Run with a live client first to populate the cache."
        )
        raise KeyError(msg)

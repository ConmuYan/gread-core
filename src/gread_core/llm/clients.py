"""LLM client abstraction: protocol, OpenAI adapter, and replay client."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    def complete(self, prompt: str) -> str:
        """Return raw string completion for *prompt*."""
        ...

    def complete_batch(self, prompts: list[str]) -> list[str]:
        ...


class OpenAIClient:
    """Thin wrapper around the OpenAI ChatCompletion API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 2,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            msg = (
                "openai package is required for OpenAIClient. "
                "Install with: pip install openai"
            )
            raise ImportError(msg) from exc

        resolved_api_key = (
            api_key
            or os.getenv("GREAD_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("MODELSCOPE_API_KEY")
            or os.getenv("MODELSCOPE_SDK_TOKEN")
        )
        resolved_base_url = (
            base_url
            or os.getenv("GREAD_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("MODELSCOPE_BASE_URL")
        )
        if resolved_api_key and resolved_base_url:
            self._client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
            )
        elif resolved_api_key:
            self._client = OpenAI(api_key=resolved_api_key)
        elif resolved_base_url:
            self._client = OpenAI(base_url=resolved_base_url)
        else:
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

    def complete_batch(self, prompts: list[str]) -> list[str]:
        return [self.complete(prompt) for prompt in prompts]


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

    def complete_batch(self, prompts: list[str]) -> list[str]:
        return [self._STUB_RESPONSE for _ in prompts]


class LocalTransformersClient:
    def __init__(
        self,
        model_path: str,
        model: str | None = None,
        temperature: float = 0.0,
        device: str = "cpu",
        batch_size: int = 1,
        max_new_tokens: int = 384,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:
            msg = (
                "transformers package is required for LocalTransformersClient. "
                "Install with: pip install transformers safetensors accelerate"
            )
            raise ImportError(msg) from exc

        transformers_mod: Any = transformers
        self._torch = torch
        self._transformers = transformers_mod
        self._model_path = model_path
        self._model_name = model or Path(model_path).name
        self._temperature = temperature
        self._device = device
        self._batch_size = max(1, int(batch_size))
        self._max_new_tokens = max(1, int(max_new_tokens))
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        """Load tokenizer/model lazily on first generation.

        Stage 2 may need a full-graph detector forward before the LLM is used.
        Delaying local model loading lets the pipeline release detector tensors
        before putting Qwen weights on the GPU.
        """
        if self._model is not None and self._tokenizer is not None:
            return
        dtype = (
            self._torch.bfloat16
            if self._device.startswith("cuda")
            else self._torch.float32
        )
        self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()

    def complete(self, prompt: str) -> str:
        return self.complete_batch([prompt])[0]

    def complete_batch(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        responses: list[str] = []
        for start in range(0, len(prompts), self._batch_size):
            chunk = prompts[start : start + self._batch_size]
            formatted = [self._format_prompt(prompt) for prompt in chunk]
            encoded = self._tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
            )
            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
            }
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": self._temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "use_cache": True,
            }
            if self._temperature > 0:
                generate_kwargs["temperature"] = self._temperature
            with self._torch.no_grad():
                output_ids = self._model.generate(
                    **encoded,
                    **generate_kwargs,
                )
            prompt_length = int(encoded["input_ids"].shape[1])
            for row in range(output_ids.shape[0]):
                generated_ids = output_ids[row, prompt_length:]
                responses.append(
                    self._tokenizer.decode(generated_ids, skip_special_tokens=True)
                )
        return responses

    def _format_prompt(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            return prompt
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(rendered)


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

    def complete_batch(self, prompts: list[str]) -> list[str]:
        return [self.complete(prompt) for prompt in prompts]

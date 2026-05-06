from __future__ import annotations

import sys
import types

import pytest

from gread_core.llm.clients import OpenAIClient


def test_openai_client_prefers_gread_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str | None] = {}

    class FakeOpenAI:
        def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("GREAD_LLM_API_KEY", "gread-key")
    monkeypatch.setenv("GREAD_LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    monkeypatch.setenv("MODELSCOPE_API_KEY", "wrong-modelscope-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong-openai.example/v1")
    monkeypatch.setenv("MODELSCOPE_BASE_URL", "https://wrong-modelscope.example/v1")

    OpenAIClient(model="qwen-test")

    assert captured == {
        "api_key": "gread-key",
        "base_url": "https://example.com/v1",
    }

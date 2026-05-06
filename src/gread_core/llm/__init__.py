"""Offline LLM teacher for evidence rationale record generation.

LLM code is training-offline only.  Inference must never import this module.
"""

from gread_core.llm.cache import PromptCache
from gread_core.llm.clients import (
    LLMClient,
    LocalTransformersClient,
    OpenAIClient,
    ReplayClient,
)
from gread_core.llm.prompt_builder import PromptBuilder
from gread_core.llm.teacher import LLMTeacher

__all__ = [
    "LLMClient",
    "LLMTeacher",
    "LocalTransformersClient",
    "OpenAIClient",
    "PromptBuilder",
    "PromptCache",
    "ReplayClient",
]

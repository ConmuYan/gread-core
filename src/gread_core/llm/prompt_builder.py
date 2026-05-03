"""Build score-blind LLM prompts from MinimalEvidencePackage."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from gread_core.schemas.risk_taxonomy import SCORE_RELATED_IDS

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "err_generation.j2"

# Only these top-level keys are allowed in a teacher payload.
_ALLOWED_PAYLOAD_KEYS: frozenset[str] = frozenset({
    "node_id", "detector_name", "reasoning",
})

# Additional keys allowed when score_blind=False (ablation mode).
_SCORE_VISIBLE_EXTRA_KEYS: frozenset[str] = frozenset({
    "prediction_score", "calibration",
})


class PromptBuilder:
    """Render a prompt from MEP reasoning payloads.

    When ``score_blind=True`` (default, main method), the payload is
    validated to exclude any score-related tokens.  When
    ``score_blind=False`` (ablation), score tokens are allowed through.
    """

    def __init__(
        self,
        template_dir: str | Path | None = None,
        score_blind: bool = True,
    ) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir or _TEMPLATE_DIR)),
            keep_trailing_newline=True,
        )
        self._template = self._env.get_template(_TEMPLATE_NAME)
        self._score_blind = score_blind

    def build(self, teacher_payload: dict[str, object]) -> str:
        """Build a prompt string from a MEP ``to_teacher_payload()`` dict.

        When score_blind=True the payload must contain only ``node_id``,
        ``detector_name``, and ``reasoning`` — never ``calibration`` or
        ``prediction_score``.

        When score_blind=False, ``prediction_score`` and ``calibration``
        are allowed (ablation mode).

        Raises:
            ValueError: If payload contains forbidden keys or score tokens
                (only when score_blind=True).
        """
        _validate_payload(teacher_payload, score_blind=self._score_blind)
        payload_json = json.dumps(teacher_payload, indent=2, ensure_ascii=False)
        return self._template.render(reasoning_payload_json=payload_json)


def _validate_payload(payload: dict[str, object], score_blind: bool = True) -> None:
    """Validate that a teacher payload is score-blind and schema-clean.

    When ``score_blind=True`` (default, main method), raises ValueError if:
    - Payload contains keys outside _ALLOWED_PAYLOAD_KEYS (e.g. calibration).
    - Any string value contains a score-related token.

    When ``score_blind=False`` (ablation), ``prediction_score`` and
    ``calibration`` keys are allowed and score-token checks are skipped.
    """
    allowed = (
        _ALLOWED_PAYLOAD_KEYS
        if score_blind
        else _ALLOWED_PAYLOAD_KEYS | _SCORE_VISIBLE_EXTRA_KEYS
    )
    extra_keys = set(payload.keys()) - allowed
    if extra_keys:
        raise ValueError(
            f"Teacher payload contains forbidden keys: {extra_keys}. "
            f"Allowed: {allowed}"
        )
    if score_blind:
        # Recursively check for score-related tokens in string values
        _check_no_score_tokens(payload, path="root")


def _check_no_score_tokens(obj: object, path: str) -> None:
    """Recursively verify no score-related tokens appear in string values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_score_tokens(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_score_tokens(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        lower = obj.lower().replace(" ", "_")
        for token in SCORE_RELATED_IDS:
            if token in lower:
                raise ValueError(
                    f"Score-related token '{token}' found at {path}: '{obj}'"
                )

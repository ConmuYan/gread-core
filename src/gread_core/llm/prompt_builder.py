"""Build score-blind LLM prompts from MinimalEvidencePackage."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "err_generation.j2"


class PromptBuilder:
    """Render a score-blind prompt from MEP reasoning payloads."""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir or _TEMPLATE_DIR)),
            keep_trailing_newline=True,
        )
        self._template = self._env.get_template(_TEMPLATE_NAME)

    def build(self, teacher_payload: dict[str, object]) -> str:
        """Build a prompt string from a MEP ``to_teacher_payload()`` dict.

        The payload must contain only ``node_id``, ``detector_name``, and
        ``reasoning`` — never ``calibration`` or ``prediction_score``.
        """
        payload_json = json.dumps(teacher_payload, indent=2, ensure_ascii=False)
        return self._template.render(reasoning_payload_json=payload_json)

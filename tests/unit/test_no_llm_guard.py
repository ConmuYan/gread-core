"""Tests for the no-LLM guard module.

Validates runtime enforcement and static analysis of forbidden imports.
"""

import sys
import tempfile
from pathlib import Path

import pytest

from gread_core.inference.no_llm_guard import (
    FORBIDDEN_IMPORTS,
    assert_no_forbidden_imports,
    check_inference_safety,
)


class TestAssertNoForbiddenImports:
    def test_passes_when_clean(self) -> None:
        """No forbidden modules loaded in a clean test environment."""
        # Ensure none of the forbidden modules are loaded
        for mod in FORBIDDEN_IMPORTS:
            sys.modules.pop(mod, None)
        assert_no_forbidden_imports()

    def test_detects_loaded_module(self) -> None:
        """Detects a forbidden module that is currently loaded."""
        fake_key = "openai"
        old = sys.modules.get(fake_key)
        try:
            sys.modules[fake_key] = type(sys)("openai")  # type: ignore[assignment]
            with pytest.raises(RuntimeError, match="openai"):
                assert_no_forbidden_imports()
        finally:
            if old is None:
                sys.modules.pop(fake_key, None)
            else:
                sys.modules[fake_key] = old


class TestCheckInferenceSafety:
    def test_clean_file(self) -> None:
        """File with no forbidden imports passes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import torch\nimport numpy as np\n")
            f.flush()
            violations = check_inference_safety(f.name)
        Path(f.name).unlink()
        assert violations == []

    def test_detects_openai_import(self) -> None:
        """Detects 'import openai'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import openai\n")
            f.flush()
            violations = check_inference_safety(f.name)
        Path(f.name).unlink()
        assert len(violations) == 1
        assert "openai" in violations[0]

    def test_detects_from_import(self) -> None:
        """Detects 'from gread_core.llm import teacher'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("from gread_core.llm import teacher\n")
            f.flush()
            violations = check_inference_safety(f.name)
        Path(f.name).unlink()
        assert len(violations) == 1
        assert "gread_core.llm" in violations[0]

    def test_detects_httpx(self) -> None:
        """Detects 'import httpx'."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import httpx\n")
            f.flush()
            violations = check_inference_safety(f.name)
        Path(f.name).unlink()
        assert len(violations) == 1
        assert "httpx" in violations[0]

    def test_inference_predictor_is_clean(self) -> None:
        """The actual inference predictor module must be LLM-free."""
        base = Path(__file__).resolve().parents[2] / "src" / "gread_core"
        predictor_path = base / "inference" / "predictor.py"
        violations = check_inference_safety(str(predictor_path))
        assert violations == []

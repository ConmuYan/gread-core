import ast
from pathlib import Path

FORBIDDEN = ["openai", "anthropic", "gread_core.llm", "requests", "httpx"]

_SCAN_DIRS = ["src/gread_core/inference", "src/gread_core/models"]


def test_no_llm_imports_in_inference_or_models() -> None:
    for directory in _SCAN_DIRS:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        for pyfile in dir_path.rglob("*.py"):
            if pyfile.name == "no_llm_guard.py":
                continue
            source = pyfile.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for f in FORBIDDEN:
                            assert f not in alias.name, f"LLM import in {pyfile}: {alias.name}"
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for f in FORBIDDEN:
                        assert f not in node.module, f"LLM import in {pyfile}: {node.module}"

import ast
from pathlib import Path

FORBIDDEN_IMPORTS = ["openai", "anthropic", "gread_core.llm", "requests", "httpx"]


def test_inference_imports_no_llm() -> None:
    inference_dir = Path("src/gread_core/inference")
    if not inference_dir.exists():
        return
    for pyfile in inference_dir.rglob("*.py"):
        if pyfile.name == "no_llm_guard.py":
            continue
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in FORBIDDEN_IMPORTS:
                        assert forbidden not in alias.name, f"{pyfile}: imports {forbidden}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in FORBIDDEN_IMPORTS:
                    assert forbidden not in node.module, f"{pyfile}: imports from {forbidden}"

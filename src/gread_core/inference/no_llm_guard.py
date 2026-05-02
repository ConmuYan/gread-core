import ast
import sys
from pathlib import Path

FORBIDDEN_IMPORTS: list[str] = [
    "openai",
    "anthropic",
    "gread_core.llm",
    "requests",
    "httpx",
]


def assert_no_forbidden_imports() -> None:
    """Raise RuntimeError if any forbidden module is currently loaded."""
    loaded = set(sys.modules.keys())
    for forbidden in FORBIDDEN_IMPORTS:
        if forbidden in loaded:
            msg = f"Forbidden module '{forbidden}' is loaded in inference context"
            raise RuntimeError(msg)


def check_inference_safety(module_path: str) -> list[str]:
    """Scan a .py file for forbidden import statements. Returns list of violations."""
    source = Path(module_path).read_text()
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in FORBIDDEN_IMPORTS:
                    if alias.name == forbidden or alias.name.startswith(forbidden + "."):
                        violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for forbidden in FORBIDDEN_IMPORTS:
                if node.module == forbidden or node.module.startswith(forbidden + "."):
                    violations.append(f"line {node.lineno}: from {node.module} import ...")
    return violations

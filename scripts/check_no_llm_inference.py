"""Static analysis: ensure inference and model paths import no LLM/network code."""
import sys
from pathlib import Path

FORBIDDEN_IMPORTS = ["openai", "anthropic", "gread_core.llm", "requests", "httpx"]

exit_code = 0
scan_dirs = ["src/gread_core/inference", "src/gread_core/models"]

for directory in scan_dirs:
    dir_path = Path(directory)
    if not dir_path.exists():
        continue
    for path in dir_path.rglob("*.py"):
        if path.name == "no_llm_guard.py":
            continue
        text = path.read_text()
        for token in FORBIDDEN_IMPORTS:
            if token in text:
                print(f"LLM IMPORT: {token} in {path}")
                exit_code = 1

sys.exit(exit_code)

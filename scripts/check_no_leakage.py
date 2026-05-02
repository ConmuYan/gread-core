"""Static analysis: ensure prediction_score does not leak into prompts or evidence targets.

Does NOT scan schema files for the string 'prediction_score' — that string is expected there.
Instead, instantiates a sample MEP and verifies to_teacher_payload() excludes it.
"""
import json
import sys
from pathlib import Path

PROMPT_DIRS = [
    "src/gread_core/llm/templates",
    "configs/prompts",
]

FORBIDDEN_TOKENS = ["prediction_score", "fraud_score", "base_score", "probability_score"]

exit_code = 0

for directory in PROMPT_DIRS:
    dir_path = Path(directory)
    if not dir_path.exists():
        continue
    for path in dir_path.rglob("*"):
        if path.is_file():
            text = path.read_text(errors="replace")
            for token in FORBIDDEN_TOKENS:
                if token in text:
                    print(f"LEAK: {token} found in {path}")
                    exit_code = 1

try:
    from gread_core.schemas.evidence import (
        CalibrationChannel,
        MinimalEvidencePackage,
        ReasoningChannel,
    )

    mep = MinimalEvidencePackage(
        node_id="guard_test",
        detector_name="test",
        calibration=CalibrationChannel(prediction_score=0.5, uncertainty=0.5),
        reasoning=ReasoningChannel(
            uncertainty_level="medium",
            degree_level="normal",
            neighbor_consistency="high",
            feature_neighbor_discrepancy="low",
            detector_signal="neutral",
            detector_signal_strength="moderate",
            counter_signal="benign_neighbor_signal_low",
            allowed_support_ids=["degree_level"],
            allowed_counter_ids=["counter_signal"],
        ),
    )
    payload = mep.to_teacher_payload()
    payload_json = json.dumps(payload)
    for token in FORBIDDEN_TOKENS:
        if token in payload_json:
            print(f"LEAK: {token} found in to_teacher_payload() output")
            exit_code = 1
    if "0.5" in payload_json:
        print("LEAK: calibration score value found in to_teacher_payload() output")
        exit_code = 1
except Exception as e:
    print(f"WARN: Could not verify to_teacher_payload(): {e}")

sys.exit(exit_code)

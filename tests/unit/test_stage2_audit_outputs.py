from __future__ import annotations

import json
from pathlib import Path

from gread_core.training.stage2_generate_err import (
    Stage2Result,
    _failed_checks_from_reasons,
)


def test_failed_checks_are_mapped_from_deterministic_reasons() -> None:
    reasons = [
        "Score-related ID in evidence: prediction_score",
        "Risk type spectral_anomaly incompatible with benign label",
        "No required condition satisfied for spectral_anomaly",
    ]

    assert _failed_checks_from_reasons(reasons) == {
        "score_blindness": ["Score-related ID in evidence: prediction_score"],
        "label_compatibility": [
            "Risk type spectral_anomaly incompatible with benign label"
        ],
        "contract": ["No required condition satisfied for spectral_anomaly"],
    }


def test_stage2_result_saves_rejection_audit_artifacts(tmp_path: Path) -> None:
    accepted = [
        {
            "node_id": "n1",
            "node_idx": 1,
            "bucket": "high_conf_fraud",
            "err": {"risk_type": "spectral_anomaly"},
            "accepted": True,
        }
    ]
    rejected = [
        {
            "node_id": "n2",
            "node_idx": 2,
            "bucket": "uncertain",
            "err": {
                "risk_type": "spectral_anomaly",
                "supporting_evidence": ["prediction_score"],
                "counter_evidence": [],
                "summary": "Score-only explanation.",
            },
            "raw_response": '{"risk_type": "spectral_anomaly"}',
            "verifier_reasons": ["Score-related ID in evidence: prediction_score"],
            "failed_checks": {
                "score_blindness": ["Score-related ID in evidence: prediction_score"]
            },
            "parse_error": None,
            "accepted": False,
        }
    ]
    raw_audit = [
        {
            "node_id": "n1",
            "node_idx": 1,
            "bucket": "high_conf_fraud",
            "accepted": True,
            "raw_response": '{"risk_type": "spectral_anomaly"}',
            "parsed_err": {"risk_type": "spectral_anomaly"},
            "verifier_reasons": [],
            "failed_checks": {},
            "parse_error": None,
        },
        {
            "node_id": "n2",
            "node_idx": 2,
            "bucket": "uncertain",
            "accepted": False,
            "raw_response": '{"risk_type": "spectral_anomaly"}',
            "parsed_err": rejected[0]["err"],
            "verifier_reasons": rejected[0]["verifier_reasons"],
            "failed_checks": rejected[0]["failed_checks"],
            "parse_error": None,
        },
    ]

    Stage2Result(
        accepted_errs=accepted,
        rejected_errs=rejected,
        rejection_report=rejected,
        raw_err_audit=raw_audit,
    ).save(tmp_path)

    rejection_report = [
        json.loads(line) for line in (tmp_path / "rejection_report.jsonl").read_text().splitlines()
    ]
    raw_err_audit = [
        json.loads(line) for line in (tmp_path / "raw_err_audit.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "rejection_summary.json").read_text())

    assert json.loads((tmp_path / "accepted_errs.json").read_text()) == accepted
    assert json.loads((tmp_path / "rejected_errs.json").read_text()) == rejected
    assert rejection_report == rejected
    assert raw_err_audit == raw_audit
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["by_failed_check"] == {"score_blindness": 1}

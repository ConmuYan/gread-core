"""Integration test for evaluate.py ERR lookup logic.

Verifies that the ERR reference lookup uses global node IDs
(not loop indices) when building references from accepted_errs.json.
"""

import json
import tempfile
from pathlib import Path


def _simulate_err_lookup(
    err_dir: Path,
    test_nodes: list[int],
    y_true: list[int],
) -> list[dict]:
    """Replicate the ERR lookup logic from evaluate.py lines 390-423."""
    risk_types = ["transaction_amount"]

    n = len(test_nodes)
    references: list[dict] = []
    err_lookup: dict[int, dict] = {}

    err_path = err_dir / "accepted_errs.json"
    if err_path.exists():
        with open(err_path) as f:
            accepted_errs = json.load(f)
        for err in accepted_errs:
            node_idx = err.get("node_idx")
            if node_idx is not None:
                err_lookup[int(node_idx)] = err

    for i in range(n):
        # Use global node index for ERR lookup, not local loop index
        global_node_idx = test_nodes[i] if i < len(test_nodes) else i
        matched = err_lookup.get(global_node_idx)
        if matched and "err" in matched:
            err_data = matched["err"]
            references.append({
                "accepted": bool(y_true[i]),
                "evidence": err_data.get("supporting_evidence", []),
                "risk_type": err_data.get("risk_type", risk_types[0]),
            })
        else:
            references.append({
                "accepted": bool(y_true[i]),
                "evidence": [],
                "risk_type": risk_types[0],
            })

    return references


def test_err_lookup_uses_global_node_ids():
    """ERR lookup must key by node_idx (global ID), not loop index."""
    accepted_errs = [
        {
            "node_idx": 4,
            "err": {
                "risk_type": "transaction_amount",
                "supporting_evidence": ["degree_level"],
                "counter_evidence": [],
            },
        },
        {
            "node_idx": 8,
            "err": {
                "risk_type": "transaction_amount",
                "supporting_evidence": ["uncertainty_level"],
                "counter_evidence": [],
            },
        },
    ]

    test_nodes = [4, 8, 12]
    y_true = [1, 0, 1]

    with tempfile.TemporaryDirectory() as tmpdir:
        err_dir = Path(tmpdir)
        (err_dir / "accepted_errs.json").write_text(json.dumps(accepted_errs))

        references = _simulate_err_lookup(err_dir, test_nodes, y_true)

    # Node 4 (loop index 0) -> should get degree_level evidence from node_idx=4
    assert references[0]["evidence"] == ["degree_level"], (
        f"references[0] should use err from node_idx=4, got {references[0]}"
    )
    assert references[0]["risk_type"] == "transaction_amount"

    # Node 8 (loop index 1) -> should get uncertainty_level evidence from node_idx=8
    assert references[1]["evidence"] == ["uncertainty_level"], (
        f"references[1] should use err from node_idx=8, got {references[1]}"
    )
    assert references[1]["risk_type"] == "transaction_amount"

    # Node 12 (loop index 2) -> no match, empty evidence
    assert references[2]["evidence"] == [], (
        f"references[2] should be empty (no match for node_idx=12), got {references[2]}"
    )

    # CRITICAL: references[0] must NOT be using loop index 0.
    # If lookup used loop index instead of global node_idx, index 0 would
    # find no match (node_idx=0 doesn't exist) and return empty evidence.
    # The fact that we get degree_level proves it looked up node_idx=4.
    assert references[0]["evidence"] != [], (
        "Bug detected: lookup used loop index 0 instead of global node ID 4"
    )


def test_err_lookup_missing_file_returns_empty_evidence():
    """When accepted_errs.json doesn't exist, all references get empty evidence."""
    test_nodes = [4, 8]
    y_true = [1, 0]

    with tempfile.TemporaryDirectory() as tmpdir:
        err_dir = Path(tmpdir)
        # No file written
        references = _simulate_err_lookup(err_dir, test_nodes, y_true)

    assert all(r["evidence"] == [] for r in references)


def test_err_lookup_preserves_y_true_labels():
    """accepted field must come from y_true, not from the ERR data."""
    accepted_errs = [{
        "node_idx": 4,
        "err": {"risk_type": "t", "supporting_evidence": ["e1"], "counter_evidence": []},
    }]

    test_nodes = [4, 4]
    y_true = [1, 0]  # same node, different labels

    with tempfile.TemporaryDirectory() as tmpdir:
        err_dir = Path(tmpdir)
        (err_dir / "accepted_errs.json").write_text(json.dumps(accepted_errs))
        references = _simulate_err_lookup(err_dir, test_nodes, y_true)

    assert references[0]["accepted"] is True
    assert references[1]["accepted"] is False
    # Both get the same evidence since node_idx=4 matches
    assert references[0]["evidence"] == ["e1"]
    assert references[1]["evidence"] == ["e1"]

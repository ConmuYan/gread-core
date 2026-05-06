from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from gread_core.cli.aggregate_stage1 import aggregate_group
from gread_core.data.loaders import load_tiny_graph
from gread_core.training.stage1_reporting import (
    apply_configured_split,
    resolve_stage1_split_config,
    save_stage1_artifacts,
)
from gread_core.training.stage1_train_detector import train_detector

try:
    from torch_geometric.data import Data

    from gread_core.detectors.pyg_gnn import GCNDetector

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYG_AVAILABLE, reason="torch_geometric not available")


def test_resolve_stage1_split_config_defaults() -> None:
    split_cfg = resolve_stage1_split_config({})
    assert split_cfg["ratios"] == [0.7, 0.1, 0.2]
    assert split_cfg["stratified"] is False


def test_apply_configured_split_respects_ratios() -> None:
    data = Data(x=torch.randn(100, 8), y=torch.randint(0, 2, (100,)))
    config = {"data": {"split": {"ratios": [0.6, 0.2, 0.2], "stratified": True}}}
    split_data = apply_configured_split(data, config, seed=7)
    train_count = int(split_data.train_mask.sum().item())
    val_count = int(split_data.val_mask.sum().item())
    test_count = int(split_data.test_mask.sum().item())
    assert train_count + val_count + test_count == 100
    assert abs(train_count - 60) <= 1
    assert abs(val_count - 20) <= 1
    assert abs(test_count - 20) <= 1


def test_save_stage1_artifacts_and_aggregate(tmp_path: Path) -> None:
    config = {
        "data": {"split": {"ratios": [0.7, 0.1, 0.2], "stratified": False}},
        "stage1": {
            "epochs": 1,
            "lr": 0.01,
            "weight_decay": 0.0005,
            "log_every": 1,
            "save_every": 1,
            "tsne": {
                "enabled": True,
                "max_points": 20,
                "perplexity": 5.0,
                "iterations": 5,
            },
        },
    }
    data = load_tiny_graph(num_nodes=30, num_features=16, seed=42)
    detector = GCNDetector(in_channels=16, hidden_channels=8)
    detector = train_detector(detector, data, config)

    run_root_a = tmp_path / "tiny" / "gcn" / "seed_42"
    summary_path_a = save_stage1_artifacts(
        run_root_a,
        detector,
        data,
        config,
        dataset="tiny",
        detector_name="gcn",
        seed=42,
    )
    assert summary_path_a.exists()
    assert (run_root_a / "stage1" / "tsne_test.png").exists()
    assert (run_root_a / "stage1" / "predictions_test.npz").exists()

    summary_a = json.loads(summary_path_a.read_text())
    summary_b = json.loads(summary_path_a.read_text())
    summary_b["seed"] = 123
    summary_b["splits"]["test"]["auprc"] = summary_a["splits"]["test"]["auprc"] + 0.2
    summary_b["training"]["best_epoch"] = summary_a["training"]["best_epoch"] + 1

    run_root_b = tmp_path / "tiny" / "gcn" / "seed_123" / "stage1"
    run_root_b.mkdir(parents=True, exist_ok=True)
    metric_path_b = run_root_b / "metrics_summary.json"
    metric_path_b.write_text(json.dumps(summary_b, indent=2))

    payload = aggregate_group([summary_path_a, metric_path_b])
    assert payload["dataset"] == "tiny"
    assert payload["detector"] == "gcn"
    assert payload["num_runs"] == 2
    assert payload["splits"]["test"]["auprc"]["mean"] == pytest.approx(
        (summary_a["splits"]["test"]["auprc"] + summary_b["splits"]["test"]["auprc"]) / 2.0
    )

"""Tests for experiment reproducibility and metadata infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from gread_core.experiment.registry import ExperimentRegistry
from gread_core.experiment.seed import set_seed


def test_set_seed_deterministic() -> None:
    """set_seed(42) then torch.randn(10) twice gives identical results."""
    set_seed(42)
    first = torch.randn(10)

    set_seed(42)
    second = torch.randn(10)

    assert torch.equal(first, second)


def test_config_hash_stable() -> None:
    """Same config dict always produces the same hash."""
    config = {"lr": 0.01, "hidden_dim": 128, "epochs": 50, "layers": [64, 32]}
    registry1 = ExperimentRegistry(
        experiment_id="exp1",
        config=config,
        config_path="configs/base.yaml",
        dataset="yelp",
        seed=42,
        output_dir="/tmp/test_hash",
    )
    registry2 = ExperimentRegistry(
        experiment_id="exp1",
        config=config,
        config_path="configs/base.yaml",
        dataset="yelp",
        seed=42,
        output_dir="/tmp/test_hash",
    )
    assert registry1.config_hash == registry2.config_hash
    assert len(registry1.config_hash) == 12


def test_registry_writes_manifest(tmp_path: Path) -> None:
    """write_manifest() creates a JSON file with all required fields."""
    config = {"lr": 0.01, "hidden_dim": 128}
    registry = ExperimentRegistry(
        experiment_id="test_exp",
        config=config,
        config_path="configs/test.yaml",
        dataset="amazon",
        seed=7,
        output_dir=tmp_path,
    )
    manifest_path = registry.write_manifest()
    assert manifest_path.exists()
    assert manifest_path.name == "experiment_test_exp.json"

    with open(manifest_path) as f:
        data = json.load(f)
    assert data["experiment_id"] == "test_exp"
    assert data["dataset"] == "amazon"
    assert data["seed"] == 7


def test_registry_manifest_fields(tmp_path: Path) -> None:
    """Manifest contains all required fields."""
    config = {"lr": 0.005}
    registry = ExperimentRegistry(
        experiment_id="field_check",
        config=config,
        config_path="configs/fields.yaml",
        dataset="tfinance",
        seed=123,
        output_dir=tmp_path,
    )
    registry.write_manifest()

    manifest_path = tmp_path / "manifests" / "experiment_field_check.json"
    with open(manifest_path) as f:
        data = json.load(f)

    required_fields = [
        "experiment_id",
        "git_commit",
        "config_hash",
        "dataset",
        "seed",
        "created_at",
        "software_versions",
        "split_hash",
        "contract_version",
        "detector_checkpoint",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    assert "python" in data["software_versions"]
    assert "torch" in data["software_versions"]

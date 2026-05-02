"""Tests for reproducibility config files.

Validates that all dataset, detector, experiment, and ablation configs
load correctly and satisfy project invariants (e.g. paper_warning presence,
experimental features disabled in main configs).
"""

import glob
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ── Dataset configs ──────────────────────────────────────────────────────


def test_dataset_configs_load():
    """Every config in configs/datasets/ loads as valid YAML."""
    pattern = str(REPO_ROOT / "configs" / "datasets" / "*.yaml")
    paths = glob.glob(pattern)
    assert paths, f"No dataset configs found matching {pattern}"
    for p in paths:
        data = _load_yaml(p)
        assert isinstance(data, dict), f"{p} did not parse as a dict"
        assert "dataset" in data, f"{p} missing 'dataset' key"


# ── Detector configs ─────────────────────────────────────────────────────


def test_detector_configs_load():
    """Every config in configs/detectors/ loads as valid YAML."""
    pattern = str(REPO_ROOT / "configs" / "detectors" / "*.yaml")
    paths = glob.glob(pattern)
    assert paths, f"No detector configs found matching {pattern}"
    for p in paths:
        data = _load_yaml(p)
        assert isinstance(data, dict), f"{p} did not parse as a dict"
        assert "type" in data, f"{p} missing 'type' key"


# ── Ablation configs ────────────────────────────────────────────────────


def test_ablation_configs_load():
    """Every ablation config in configs/experiments/ loads as valid YAML."""
    pattern = str(REPO_ROOT / "configs" / "experiments" / "ablation_*.yaml")
    paths = glob.glob(pattern)
    assert paths, f"No ablation configs found matching {pattern}"
    for p in paths:
        data = _load_yaml(p)
        assert isinstance(data, dict), f"{p} did not parse as a dict"


def test_ablation_configs_have_paper_warning():
    """Every ablation config must declare a paper_warning key."""
    pattern = str(REPO_ROOT / "configs" / "experiments" / "ablation_*.yaml")
    paths = glob.glob(pattern)
    assert paths, f"No ablation configs found matching {pattern}"
    for p in paths:
        data = _load_yaml(p)
        assert "paper_warning" in data, (
            f"{os.path.basename(p)} missing 'paper_warning' key"
        )
        assert isinstance(data["paper_warning"], str) and data["paper_warning"], (
            f"{os.path.basename(p)} has empty paper_warning"
        )


# ── Main config invariants ──────────────────────────────────────────────


def test_main_configs_no_experimental():
    """main_gcn_tiny.yaml must have all experimental features disabled."""
    path = REPO_ROOT / "configs" / "experiments" / "main_gcn_tiny.yaml"
    data = _load_yaml(str(path))
    experimental = data.get("experimental", {})
    for feature, settings in experimental.items():
        assert settings.get("enabled") is False, (
            f"Experimental feature '{feature}' is not disabled in main_gcn_tiny.yaml"
        )

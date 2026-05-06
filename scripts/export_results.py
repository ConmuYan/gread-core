"""export_results.py — Export experiment metrics to CSV tables.

Reads all artifacts/*/metrics/evaluation_results.json files and produces:
  - artifacts/tables/main_table.csv  (experiment, dataset, detector, AUC, AUPRC, F1)
  - artifacts/tables/ablation_table.csv  (experiment, paper_warning, AUC, AUPRC, F1, acceptance)

Usage:
    python scripts/export_results.py
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
TABLES_DIR = ARTIFACTS_DIR / "tables"

# Metrics we extract from evaluation_results.json -> detection sub-dict
DETECTION_KEYS = ("auc", "auprc", "f1")
# Metrics from evaluation_results.json -> reasoning sub-dict
REASONING_KEYS = ("acceptance_rate", "evidence_f1", "risk_type_accuracy")


def _find_result_files() -> list[Path]:
    """Discover publishable evaluation result files."""
    return sorted(
        path
        for path in ARTIFACTS_DIR.glob("*/*/evaluation_results.json")
        if path.parent.name == "metrics" or path.parent.name.startswith("real_metrics")
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(path) as f:
        return json.load(f)


def _get_nested(data: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Safely traverse nested dicts and return a float value."""
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    if current is None:
        return default
    return float(current)


def _infer_experiment_info(experiment_dir_name: str) -> dict[str, str]:
    """Infer dataset, detector, and ablation status from the experiment directory name.

    Heuristic rules:
      - Names starting with 'ablation_' are ablation experiments.
      - Otherwise, try to extract dataset/detector from the name (e.g. main_gcn_tiny).
    """
    info: dict[str, str] = {"experiment": experiment_dir_name, "is_ablation": "false"}

    if experiment_dir_name.startswith("ablation_"):
        info["is_ablation"] = "true"
        # Derive a human-readable paper_warning from the config
        info["paper_warning"] = experiment_dir_name.replace("ablation_", "") + "_ablation"
        info["dataset"] = "tiny"
        info["detector"] = "gcn"
    elif experiment_dir_name.startswith("main_"):
        # e.g. main_gcn_tiny or main_gat_yahoo
        parts = experiment_dir_name.split("_")
        if len(parts) >= 3:
            info["detector"] = parts[1]
            info["dataset"] = "_".join(parts[2:])
        else:
            info["dataset"] = "unknown"
            info["detector"] = "unknown"
    elif experiment_dir_name == "smoke":
        info["dataset"] = "tiny"
        info["detector"] = "gcn"
    else:
        info["dataset"] = "unknown"
        info["detector"] = "unknown"

    return info


def _result_context(result_path: Path, data: dict[str, Any]) -> dict[str, str]:
    """Read experiment/dataset/detector context from result metadata first."""
    experiment_dir = result_path.parent.parent
    info = _infer_experiment_info(experiment_dir.name)
    dataset = data.get("dataset") or info.get("dataset", "unknown")
    detector = data.get("detector") or info.get("detector", "unknown")
    info["dataset"] = str(dataset)
    info["detector"] = str(detector)
    return info


def _is_publishable_real_result(data: dict[str, Any], info: dict[str, str]) -> bool:
    """Return True when a result is real and has non-placeholder routing metadata."""
    dataset = info.get("dataset", "unknown")
    detector = info.get("detector", "unknown")
    return (
        data.get("evaluation_mode") == "real"
        and dataset not in {"", "unknown", "synthetic"}
        and detector not in {"", "unknown", "synthetic"}
    )


def _try_read_config_warning(experiment_dir: Path) -> str:
    """Try to read paper_warning from the corresponding YAML config."""
    # The config is typically at configs/experiments/<experiment_dir_name>.yaml
    config_path = Path("configs/experiments") / f"{experiment_dir.name}.yaml"
    if not config_path.exists():
        return ""
    try:
        import yaml  # type: ignore[import-untyped]

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            return str(cfg.get("paper_warning", ""))
    except Exception:
        pass
    return ""


def build_main_table(results: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, str]]:
    """Build rows for the main results table."""
    rows: list[dict[str, str]] = []
    for result_path, data in results:
        info = _result_context(result_path, data)

        # Skip ablation experiments in the main table
        if info["is_ablation"] == "true":
            continue
        if not _is_publishable_real_result(data, info):
            continue

        row: dict[str, str] = {
            "experiment": info["experiment"],
            "dataset": info.get("dataset", "unknown"),
            "detector": info.get("detector", "unknown"),
        }

        detection = data.get("detection", {})
        for key in DETECTION_KEYS:
            row[key.upper() if key != "f1" else "F1"] = f"{detection.get(key, 0.0):.4f}"

        rows.append(row)

    return rows


def build_ablation_table(results: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, str]]:
    """Build rows for the ablation results table."""
    rows: list[dict[str, str]] = []
    for result_path, data in results:
        experiment_dir = result_path.parent.parent
        info = _result_context(result_path, data)

        # Only include ablation experiments
        if info["is_ablation"] != "true":
            continue
        if not _is_publishable_real_result(data, info):
            continue

        # Try to get paper_warning from config file first, fall back to inferred
        paper_warning = _try_read_config_warning(experiment_dir)
        if not paper_warning:
            paper_warning = info.get("paper_warning", "")

        row: dict[str, str] = {
            "experiment": info["experiment"],
            "paper_warning": paper_warning,
        }

        detection = data.get("detection", {})
        for key in DETECTION_KEYS:
            row[key.upper() if key != "f1" else "F1"] = f"{detection.get(key, 0.0):.4f}"

        reasoning = data.get("reasoning", {})
        row["acceptance"] = f"{reasoning.get('acceptance_rate', 0.0):.4f}"

        rows.append(row)

    return rows


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    """Entry point: discover results, build tables, write CSVs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result_files = _find_result_files()
    if not result_files:
        logger.warning("No evaluation_results.json found under artifacts/*/metrics/")
        return

    logger.info("Found %d result file(s)", len(result_files))

    # Load all results
    results: list[tuple[Path, dict[str, Any]]] = []
    for path in result_files:
        try:
            data = _load_json(path)
            results.append((path, data))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path, exc)

    # Main table
    main_rows = build_main_table(results)
    main_fields = ["experiment", "dataset", "detector", "AUC", "AUPRC", "F1"]
    _write_csv(TABLES_DIR / "main_table.csv", main_rows, main_fields)

    # Ablation table
    ablation_rows = build_ablation_table(results)
    ablation_fields = [
        "experiment", "paper_warning", "AUC", "AUPRC", "F1", "acceptance",
    ]
    _write_csv(TABLES_DIR / "ablation_table.csv", ablation_rows, ablation_fields)

    logger.info("Done. Tables written to %s", TABLES_DIR)


if __name__ == "__main__":
    main()

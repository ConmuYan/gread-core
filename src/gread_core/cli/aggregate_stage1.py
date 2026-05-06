from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _discover_metric_files(runs_root: Path) -> list[Path]:
    return sorted(runs_root.glob("*/ */seed_*/stage1/metrics_summary.json".replace(" ", "")))


def _numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate_split_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    split_names = sorted({name for summary in summaries for name in summary.get("splits", {})})
    aggregated: dict[str, Any] = {}
    for split_name in split_names:
        metric_names = sorted(
            {
                key
                for summary in summaries
                for key, value in summary.get("splits", {}).get(split_name, {}).items()
                if isinstance(value, (int, float))
            }
        )
        aggregated[split_name] = {
            metric_name: _numeric_summary(
                [
                    float(summary["splits"][split_name][metric_name])
                    for summary in summaries
                    if metric_name in summary.get("splits", {}).get(split_name, {})
                ]
            )
            for metric_name in metric_names
        }
    return aggregated


def _aggregate_training_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    training_keys = sorted(
        {
            key
            for summary in summaries
            for key, value in summary.get("training", {}).items()
            if isinstance(value, (int, float))
        }
    )
    aggregated: dict[str, Any] = {
        key: _numeric_summary(
            [
                float(summary["training"][key])
                for summary in summaries
                if key in summary.get("training", {})
            ]
        )
        for key in training_keys
    }
    monitors = [summary.get("training", {}).get("monitor") for summary in summaries]
    aggregated["monitor"] = monitors[0] if monitors else None
    return aggregated


def aggregate_group(metric_files: list[Path]) -> dict[str, Any]:
    summaries = [json.loads(path.read_text()) for path in metric_files]
    if not summaries:
        raise ValueError("No Stage1 metric summaries found")
    seeds = [int(summary["seed"]) for summary in summaries]
    first = summaries[0]
    return {
        "dataset": first["dataset"],
        "detector": first["detector"],
        "num_runs": len(summaries),
        "seeds": sorted(seeds),
        "split": first.get("split", {}),
        "decision_threshold": first.get("decision_threshold"),
        "training": _aggregate_training_metrics(summaries),
        "splits": _aggregate_split_metrics(summaries),
        "runs": [str(path.parent.parent) for path in metric_files],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage1 metrics across seeds")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="artifacts/stage1_rework_runs",
        help="Root directory containing dataset/detector/seed_* Stage1 runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path for the full aggregation index",
    )
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    metric_files = _discover_metric_files(runs_root)
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for metric_file in metric_files:
        seed_dir = metric_file.parent.parent
        detector_dir = seed_dir.parent
        dataset_dir = detector_dir.parent
        grouped[(dataset_dir.name, detector_dir.name)].append(metric_file)

    aggregated_index: dict[str, Any] = {"groups": []}
    for (dataset, detector), files in sorted(grouped.items()):
        payload = aggregate_group(files)
        output_path = runs_root / dataset / detector / "stage1_aggregate.json"
        output_path.write_text(json.dumps(payload, indent=2))
        aggregated_index["groups"].append(
            {
                "dataset": dataset,
                "detector": detector,
                "path": str(output_path),
                "num_runs": payload["num_runs"],
            }
        )

    output_path = Path(args.output) if args.output else runs_root / "stage1_aggregate_index.json"
    output_path.write_text(json.dumps(aggregated_index, indent=2))


if __name__ == "__main__":
    main()

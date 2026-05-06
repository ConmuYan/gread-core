#!/usr/bin/env bash
set -uo pipefail
SKIP_DONE="${SKIP_DONE:-1}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"
CONFIG="${CONFIG:-configs/experiments/stage1_rework.yaml}"
DATA_ROOT="${DATA_ROOT:-${GREAD_DATA_ROOT:-data/raw}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/stage1_rework_runs}"
DATASETS="${DATASETS:-yelpchi amazon tfinance tsocial}"
DETECTORS="${DETECTORS:-gcn gat bwgnn caregnn tree_neighbor sage pc_gnn h2gcn gin gpr_gnn}"
SEEDS="${SEEDS:-42 123 456 789 2026}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_DIR"

printf '=== Stage1 Rework Launcher ===\n'
printf 'project:   %s\n' "$PROJECT_DIR"
printf 'python:    %s\n' "$PYTHON_BIN"
printf 'config:    %s\n' "$CONFIG"
printf 'data_root: %s\n' "$DATA_ROOT"
printf 'output:    %s\n' "$OUTPUT_ROOT"
printf 'gpu_id:    %s\n' "$GPU_ID"
printf 'datasets:  %s\n' "$DATASETS"
printf 'detectors: %s\n' "$DETECTORS"
printf 'seeds:     %s\n\n' "$SEEDS"

for dataset in $DATASETS; do
  for detector in $DETECTORS; do
    for seed in $SEEDS; do
      run_dir="$OUTPUT_ROOT/$dataset/$detector/seed_$seed"
      exp_id="stage1_${dataset}_${detector}_s${seed}"
      summary="$run_dir/stage1/metrics_summary.json"
      if [[ "$SKIP_DONE" == "1" && -f "$summary" ]]; then
        echo "[Stage1] SKIP (done) dataset=$dataset detector=$detector seed=$seed"
        continue
      fi
      echo "[Stage1] dataset=$dataset detector=$detector seed=$seed gpu=$GPU_ID"
      mkdir -p "$run_dir"
      if ! CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m gread_core.cli.train_detector \
        --config "$CONFIG" \
        --dataset "$dataset" \
        --detector "$detector" \
        --output-dir "$run_dir" \
        --experiment-id "$exp_id" \
        --seed "$seed" \
        --device "$DEVICE" \
        --data-root "$DATA_ROOT"; then
        echo "[Stage1] FAIL dataset=$dataset detector=$detector seed=$seed (continuing)"
      fi
    done
  done
done

#!/usr/bin/env bash
# run_ablations.sh — Iterate over all ablation configs and run the pipeline.
#
# Usage: bash scripts/run_ablations.sh [dataset] [detector] [seed]
# Defaults: dataset=tiny, detector=gcn, seed=1
#
# Discovers configs matching configs/experiments/ablation_*.yaml and runs
# Stage 1/2/3 + evaluate for each.  Static analysis (ruff/mypy) is skipped
# for speed — run `bash scripts/run_main_table.sh` first to verify code quality.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATASET="${1:-tiny}"
DETECTOR="${2:-gcn}"
SEED="${3:-1}"

# ── Discover ablation configs ───────────────────────────────────────
CONFIGS=(configs/experiments/ablation_*.yaml)
if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "ERROR: No ablation configs found matching configs/experiments/ablation_*.yaml"
    exit 1
fi

echo "=== GReaD-Core Ablation Sweep ==="
echo "  dataset:  $DATASET"
echo "  detector: $DETECTOR"
echo "  seed:     $SEED"
echo "  configs:  ${#CONFIGS[@]} found"
echo ""

TOTAL=${#CONFIGS[@]}
PASSED=0
FAILED=0

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NUM=$((i + 1))
    EXPERIMENT_ID="$(basename "$CONFIG" .yaml)"
    OUTPUT_DIR="artifacts/${EXPERIMENT_ID}"

    echo "────────────────────────────────────────────────────────"
    echo "[$NUM/$TOTAL] $EXPERIMENT_ID"
    echo "  config: $CONFIG"
    echo "  output: $OUTPUT_DIR"
    echo ""

    # ── Stage 1: Train base detector ────────────────────────────────
    echo "  [1/4] Stage 1: Train base detector..."
    python -m gread_core.cli.train_detector \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --detector "$DETECTOR" \
        --output-dir "$OUTPUT_DIR" \
        --experiment-id "$EXPERIMENT_ID" \
        --seed "$SEED"

    # Find the latest stage1 checkpoint
    STAGE1_CKPT=$(ls -d "$OUTPUT_DIR"/stage1/epoch_* 2>/dev/null | sort | tail -1)
    if [ -z "$STAGE1_CKPT" ]; then
        echo "  ERROR: No Stage 1 checkpoint found — skipping $EXPERIMENT_ID"
        FAILED=$((FAILED + 1))
        continue
    fi

    # ── Stage 2: Generate ERRs (stub mode) ──────────────────────────
    echo "  [2/4] Stage 2: Generate ERRs..."
    python -m gread_core.cli.generate_err \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --detector "$DETECTOR" \
        --checkpoint "$STAGE1_CKPT" \
        --output-dir "$OUTPUT_DIR" \
        --experiment-id "$EXPERIMENT_ID" \
        --seed "$SEED" \
        --llm-backend stub \
        --cache-dir ".cache/llm_${EXPERIMENT_ID}"

    # ── Stage 3: Train reasoner ─────────────────────────────────────
    echo "  [3/4] Stage 3: Train reasoner..."
    python -m gread_core.cli.train_reasoner \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --detector "$DETECTOR" \
        --detector-checkpoint "$STAGE1_CKPT" \
        --err-dir "$OUTPUT_DIR/stage2" \
        --output-dir "$OUTPUT_DIR" \
        --experiment-id "$EXPERIMENT_ID" \
        --seed "$SEED"

    # Find the latest stage3 checkpoint
    STAGE3_CKPT=$(ls -d "$OUTPUT_DIR"/stage3/epoch_* 2>/dev/null | sort | tail -1)
    if [ -z "$STAGE3_CKPT" ]; then
        echo "  ERROR: No Stage 3 checkpoint found — skipping $EXPERIMENT_ID"
        FAILED=$((FAILED + 1))
        continue
    fi

    # ── Evaluate ────────────────────────────────────────────────────
    echo "  [4/4] Evaluation..."
    python -m gread_core.cli.evaluate \
        --checkpoint "$STAGE3_CKPT" \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --detector "$DETECTOR" \
        --detector-checkpoint "$STAGE1_CKPT" \
        --output "$OUTPUT_DIR/metrics" \
        --seed "$SEED"

    PASSED=$((PASSED + 1))
    echo ""
    echo "  -> $EXPERIMENT_ID complete"
    echo ""
done

echo "════════════════════════════════════════════════════════"
echo "=== Ablation sweep finished ==="
echo "  total:  $TOTAL"
echo "  passed: $PASSED"
echo "  failed: $FAILED"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED ablation(s) failed. Check logs above."
    exit 1
fi

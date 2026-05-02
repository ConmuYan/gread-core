#!/usr/bin/env bash
# run_main_table.sh — Run the full GReaD-Core pipeline for a given config.
#
# Usage: bash scripts/run_main_table.sh <config> [dataset] [detector] [seed]
# Defaults: dataset=tiny, detector=gcn, seed=1
#
# Example:
#   bash scripts/run_main_table.sh configs/experiments/main_gcn_tiny.yaml
#   bash scripts/run_main_table.sh configs/experiments/main_gcn_tiny.yaml yelp gat 42

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Arguments ────────────────────────────────────────────────────────
CONFIG="${1:?Usage: bash scripts/run_main_table.sh <config> [dataset] [detector] [seed]}"
DATASET="${2:-tiny}"
DETECTOR="${3:-gcn}"
SEED="${4:-1}"

# Derive experiment id from config filename (strip path and extension)
EXPERIMENT_ID="$(basename "$CONFIG" .yaml)"
OUTPUT_DIR="artifacts/${EXPERIMENT_ID}"

echo "=== GReaD-Core Main Table ==="
echo "  config:   $CONFIG"
echo "  dataset:  $DATASET"
echo "  detector: $DETECTOR"
echo "  seed:     $SEED"
echo "  output:   $OUTPUT_DIR"
echo ""

# ── Step 1: Static analysis ─────────────────────────────────────────
echo "[1/6] ruff check..."
ruff check .

echo "[2/6] mypy..."
mypy src

# ── Step 2: Stage 1 — Train base detector ────────────────────────────
echo ""
echo "[3/6] Stage 1: Train base detector..."
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
    echo "ERROR: No Stage 1 checkpoint found in $OUTPUT_DIR/stage1/"
    exit 1
fi
echo "  -> Stage 1 checkpoint: $STAGE1_CKPT"

# ── Step 3: Stage 2 — Generate ERRs (stub mode) ─────────────────────
echo ""
echo "[4/6] Stage 2: Generate ERRs (stub mode)..."
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

# ── Step 4: Stage 3 — Train reasoner ────────────────────────────────
echo ""
echo "[5/6] Stage 3: Train reasoner..."
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
    echo "ERROR: No Stage 3 checkpoint found in $OUTPUT_DIR/stage3/"
    exit 1
fi
echo "  -> Stage 3 checkpoint: $STAGE3_CKPT"

# ── Step 5: Evaluate ────────────────────────────────────────────────
echo ""
echo "[6/6] Evaluation..."
python -m gread_core.cli.evaluate \
    --checkpoint "$STAGE3_CKPT" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR/metrics" \
    --seed "$SEED"

echo ""
echo "=== Pipeline complete ==="
echo "Artifacts: $OUTPUT_DIR/"

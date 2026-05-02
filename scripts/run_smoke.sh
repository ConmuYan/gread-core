#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/experiments/smoke_tiny.yaml"
OUTPUT_DIR="artifacts/smoke"
SEED=1

echo "=== GReaD-Core Smoke Test ==="
echo ""

# ── Static analysis ──────────────────────────────────────────────────
echo "[1/8] ruff check..."
ruff check .

echo "[2/8] mypy..."
mypy src

echo "[3/8] unit tests..."
pytest tests/unit -v --tb=short

echo "[4/8] paper alignment tests..."
pytest tests/paper_alignment -v --tb=short

echo "[5/8] static guards..."
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py

# ── End-to-end training pipeline ─────────────────────────────────────
echo ""
echo "=== End-to-End Pipeline (tiny graph, CPU) ==="
echo ""

echo "[6/8] Stage 1: Train base detector..."
python -m gread_core.cli.train_detector \
    --config "$CONFIG" \
    --dataset tiny \
    --detector gcn \
    --output-dir "$OUTPUT_DIR" \
    --experiment-id smoke \
    --seed "$SEED" \
    --device cpu

# Find the latest stage1 checkpoint
STAGE1_CKPT=$(ls -d "$OUTPUT_DIR"/stage1/epoch_* 2>/dev/null | sort | tail -1)
if [ -z "$STAGE1_CKPT" ]; then
    echo "ERROR: No Stage 1 checkpoint found"
    exit 1
fi
echo "  -> Stage 1 checkpoint: $STAGE1_CKPT"

echo "[7/8] Stage 2: Generate ERRs (stub mode)..."
python -m gread_core.cli.generate_err \
    --config "$CONFIG" \
    --dataset tiny \
    --detector gcn \
    --checkpoint "$STAGE1_CKPT" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-id smoke \
    --seed "$SEED" \
    --device cpu \
    --llm-backend stub \
    --cache-dir .cache/llm_smoke

echo "[8/8] Stage 3: Train reasoner..."
python -m gread_core.cli.train_reasoner \
    --config "$CONFIG" \
    --dataset tiny \
    --detector gcn \
    --detector-checkpoint "$STAGE1_CKPT" \
    --err-dir "$OUTPUT_DIR/stage2" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-id smoke \
    --seed "$SEED" \
    --device cpu

# Find the latest stage3 checkpoint
STAGE3_CKPT=$(ls -d "$OUTPUT_DIR"/stage3/epoch_* 2>/dev/null | sort | tail -1)
if [ -z "$STAGE3_CKPT" ]; then
    echo "ERROR: No Stage 3 checkpoint found"
    exit 1
fi
echo "  -> Stage 3 checkpoint: $STAGE3_CKPT"

echo ""
echo "=== Evaluation ==="
python -m gread_core.cli.evaluate \
    --checkpoint "$STAGE3_CKPT" \
    --config "$CONFIG" \
    --output "$OUTPUT_DIR/metrics" \
    --seed "$SEED" \
    --device cpu

echo ""
echo "=== All smoke checks passed ==="
echo "Artifacts: $OUTPUT_DIR/"

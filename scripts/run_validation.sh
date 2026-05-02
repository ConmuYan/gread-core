#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Usage: bash scripts/run_validation.sh [dataset] [detector]
# If no args, runs all combinations.

DATASETS=${1:-"yelpchi amazon tfinance tsocial"}
DETECTORS=${2:-"gcn bwgnn"}
SEED=1
DEVICE="cuda"

for DATASET in $DATASETS; do
    for DETECTOR in $DETECTORS; do
        EXP_ID="val_${DETECTOR}_${DATASET}"
        CONFIG="configs/experiments/val_${DETECTOR}_${DATASET}.yaml"
        OUTPUT_DIR="artifacts/${EXP_ID}"
        TB_DIR="${OUTPUT_DIR}/tensorboard"

        if [ ! -f "$CONFIG" ]; then
            echo "SKIP: $CONFIG not found"
            continue
        fi

        echo ""
        echo "=== Validation: ${DETECTOR} on ${DATASET} ==="
        echo ""

        rm -rf "$OUTPUT_DIR"

        # Stage 1: Train detector
        echo "[1/4] Stage 1: Train ${DETECTOR} on ${DATASET}..."
        CUDA_VISIBLE_DEVICES=1 python -m gread_core.cli.train_detector \
            --config "$CONFIG" \
            --dataset "$DATASET" \
            --detector "$DETECTOR" \
            --output-dir "$OUTPUT_DIR" \
            --experiment-id "$EXP_ID" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --tensorboard-dir "$TB_DIR"

        STAGE1_CKPT=$(ls -d "$OUTPUT_DIR"/stage1/epoch_* 2>/dev/null | sort | tail -1)
        echo "  -> Stage 1 checkpoint: $STAGE1_CKPT"

        # Stage 2: Generate ERRs (stub mode)
        echo "[2/4] Stage 2: Generate ERRs..."
        python -m gread_core.cli.generate_err \
            --config "$CONFIG" \
            --dataset "$DATASET" \
            --detector "$DETECTOR" \
            --checkpoint "$STAGE1_CKPT" \
            --output-dir "$OUTPUT_DIR" \
            --experiment-id "$EXP_ID" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --llm-backend stub \
            --cache-dir ".cache/llm_${EXP_ID}"

        # Stage 3: Train reasoner
        echo "[3/4] Stage 3: Train reasoner..."
        python -m gread_core.cli.train_reasoner \
            --config "$CONFIG" \
            --dataset "$DATASET" \
            --detector "$DETECTOR" \
            --detector-checkpoint "$STAGE1_CKPT" \
            --err-dir "$OUTPUT_DIR/stage2" \
            --output-dir "$OUTPUT_DIR" \
            --experiment-id "$EXP_ID" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --tensorboard-dir "$TB_DIR"

        STAGE3_CKPT=$(ls -d "$OUTPUT_DIR"/stage3/epoch_* 2>/dev/null | sort | tail -1)
        echo "  -> Stage 3 checkpoint: $STAGE3_CKPT"

        # Stage 4: Evaluate
        echo "[4/4] Evaluation..."
        python -m gread_core.cli.evaluate \
            --checkpoint "$STAGE3_CKPT" \
            --config "$CONFIG" \
            --output "$OUTPUT_DIR/metrics" \
            --seed "$SEED" \
            --device "$DEVICE"

        echo ""
        echo "=== ${DETECTOR} on ${DATASET}: COMPLETE ==="
        echo "  Artifacts: $OUTPUT_DIR"
        echo "  TensorBoard: $TB_DIR"
        echo ""
    done
done

echo ""
echo "=== All validation runs complete ==="
echo "Launch TensorBoard: tensorboard --logdir artifacts/val_*/tensorboard"

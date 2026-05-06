#!/usr/bin/env bash
# run_smoke_matrix.sh — Fast per-detector smoke cells on the tiny graph.
#
# For each detector in $DETECTORS, generates a 1-epoch config derived from
# configs/experiments/smoke_tiny.yaml and runs stage1 -> stage2 (stub LLM) ->
# stage3 -> evaluate. Writes a tab-separated summary to
# $ROOT_OUT/smoke_matrix_summary.tsv and prints a pretty table at the end.
#
# Usage:
#   scripts/run_smoke_matrix.sh
#   DETECTORS="gcn gat sage" scripts/run_smoke_matrix.sh
#   DATASET=tiny ROOT_OUT=artifacts/smoke_matrix scripts/run_smoke_matrix.sh
#
# Exits non-zero if any cell fails, so CI can flag regressions.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# -- Configuration (env-overridable) ---------------------------------
DETECTORS="${DETECTORS:-gcn gat sage gin}"
DATASET="${DATASET:-tiny}"
ROOT_OUT="${ROOT_OUT:-artifacts/smoke_matrix}"
SEED="${SEED:-1}"
DATA_ROOT="${GREAD_DATA_ROOT:-data/raw}"
BASE_CONFIG="${BASE_CONFIG:-configs/experiments/smoke_tiny.yaml}"

mkdir -p "$ROOT_OUT"
SUMMARY="$ROOT_OUT/smoke_matrix_summary.tsv"
printf 'detector\tstage1\tstage2\tstage3\tevaluate\tauc\tauprc\tf1\n' > "$SUMMARY"

# -- Helpers ---------------------------------------------------------
gen_config() {
    local det="$1"
    local out="$ROOT_OUT/configs/smoke_matrix_${det}.yaml"
    mkdir -p "$(dirname "$out")"
    python - "$BASE_CONFIG" "$out" "$det" <<'PY'
import sys, yaml, pathlib
src, dst, det = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(pathlib.Path(src).read_text()) or {}
cfg.setdefault("detector", {})["type"] = det
for stage in ("stage1", "stage3"):
    cfg.setdefault(stage, {})["epochs"] = 1
    cfg[stage]["save_every"] = 1
    cfg[stage]["log_every"] = 1
cfg.setdefault("stage2", {})["llm_backend"] = "stub"
cfg["stage2"]["cache_dir"] = f".cache/llm_smoke_{det}"
cfg["stage2"]["trace_budget"] = 4
cfg.setdefault("trace_selection", {})["total_budget"] = 4
pathlib.Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

run_cell() {
    local det="$1"
    local cfg="$ROOT_OUT/configs/smoke_matrix_${det}.yaml"
    local out_dir="$ROOT_OUT/cells/${det}"
    local log="$ROOT_OUT/cells/${det}.log"
    mkdir -p "$out_dir"
    : > "$log"

    local s1="FAIL" s2="FAIL" s3="FAIL" ev="FAIL"
    local auc="-" auprc="-" f1="-"

    echo "---- [$det] stage1 ----" | tee -a "$log"
    if python -m gread_core.cli.train_detector \
            --config "$cfg" --dataset "$DATASET" --detector "$det" \
            --output-dir "$out_dir" --experiment-id "smoke_${det}" \
            --seed "$SEED" --device cpu --data-root "$DATA_ROOT" >>"$log" 2>&1; then
        s1="OK"
    fi

    local ckpt1=""
    if [ "$s1" = "OK" ]; then
        ckpt1=$(ls -d "$out_dir"/stage1/epoch_* 2>/dev/null | sort | tail -1 || true)
        [ -z "$ckpt1" ] && s1="FAIL"
    fi

    if [ "$s1" = "OK" ]; then
        echo "---- [$det] stage2 ----" | tee -a "$log"
        rm -rf ".cache/llm_smoke_${det}"
        if python -m gread_core.cli.generate_err \
                --config "$cfg" --dataset "$DATASET" --detector "$det" \
                --checkpoint "$ckpt1" --output-dir "$out_dir" \
                --experiment-id "smoke_${det}" --seed "$SEED" --device cpu \
                --data-root "$DATA_ROOT" --llm-backend stub \
                --cache-dir ".cache/llm_smoke_${det}" >>"$log" 2>&1; then
            s2="OK"
        fi
    fi

    if [ "$s2" = "OK" ]; then
        echo "---- [$det] stage3 ----" | tee -a "$log"
        if python -m gread_core.cli.train_reasoner \
                --config "$cfg" --dataset "$DATASET" --detector "$det" \
                --detector-checkpoint "$ckpt1" \
                --err-dir "$out_dir/stage2" --output-dir "$out_dir" \
                --experiment-id "smoke_${det}" --seed "$SEED" --device cpu \
                --data-root "$DATA_ROOT" >>"$log" 2>&1; then
            s3="OK"
        fi
    fi

    local ckpt3=""
    if [ "$s3" = "OK" ]; then
        ckpt3=$(ls -d "$out_dir"/stage3/epoch_* 2>/dev/null | sort | tail -1 || true)
        [ -z "$ckpt3" ] && s3="FAIL"
    fi

    if [ "$s3" = "OK" ]; then
        echo "---- [$det] evaluate ----" | tee -a "$log"
        if python -m gread_core.cli.evaluate \
                --checkpoint "$ckpt3" --config "$cfg" \
                --dataset "$DATASET" --detector "$det" \
                --detector-checkpoint "$ckpt1" \
                --err-dir "$out_dir/stage2" --output "$out_dir/metrics" \
                --seed "$SEED" --device cpu --data-root "$DATA_ROOT" >>"$log" 2>&1; then
            ev="OK"
        fi
    fi

    if [ "$ev" = "OK" ] && [ -f "$out_dir/metrics/evaluation_results.json" ]; then
        read -r auc auprc f1 < <(python - "$out_dir/metrics/evaluation_results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
det = d.get("detection", {}) if isinstance(d, dict) else {}
def fmt(v):
    try: return f"{float(v):.4f}"
    except Exception: return "-"
print(fmt(det.get("auc")), fmt(det.get("auprc")), fmt(det.get("f1")))
PY
)
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$det" "$s1" "$s2" "$s3" "$ev" "$auc" "$auprc" "$f1" >> "$SUMMARY"

    [ "$ev" = "OK" ]
}

# -- Main ------------------------------------------------------------
echo "=== GReaD-Core Smoke Matrix ==="
echo "detectors: $DETECTORS"
echo "dataset:   $DATASET   (stub LLM, 1 epoch per stage)"
echo "output:    $ROOT_OUT"
echo ""

TOTAL=0
PASSED=0
for det in $DETECTORS; do
    TOTAL=$((TOTAL + 1))
    if ! gen_config "$det"; then
        echo "  !! gen_config failed for $det"
        printf '%s\tFAIL\tFAIL\tFAIL\tFAIL\t-\t-\t-\n' "$det" >> "$SUMMARY"
        continue
    fi
    if run_cell "$det"; then
        PASSED=$((PASSED + 1))
    fi
done

echo ""
echo "=== Smoke Matrix Summary ==="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo ""
echo "passed: $PASSED / $TOTAL"

[ "$PASSED" -eq "$TOTAL" ]

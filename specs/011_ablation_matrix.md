# Ablation Matrix

## Purpose

Ablations must isolate the value of each main design choice.

## Main Configuration

Main method:

```yaml
score_blind: true
evidence_contract_verifier: true
label_compatibility: true
signed_evidence_masks: true
evidence_conditioned_reasoner: true
residual_rho: 0.1
llm_free_inference: true
diversity_trace_selection: true
```

## Required Ablations

### A1. Base Detector Only

Disable all GReaD components.

Purpose:

```text
measure base detection performance
```

### A2. Naive Heads Without LLM

Train risk/evidence heads without LLM-generated ERR.

Purpose:

```text
measure value of verified LLM rationale supervision
```

### A3. LLM ERR Without Verifier

Use LLM ERRs without Evidence Contract Verifier.

Purpose:

```text
measure verifier necessity
```

### A4. Schema-Only Verifier

Use only JSON schema and taxonomy checks.

Purpose:

```text
show Evidence Contract Verifier is stronger than schema validation
```

### A5. No Score-Blind MEP

Expose prediction score to LLM prompt.

Purpose:

```text
measure score leakage and score echo risk
```

Must be labeled:

```text
ablation only, not main method
```

### A6. No Label Compatibility

Disable label compatibility check.

Purpose:

```text
measure factual polarity protection
```

### A7. No Role Consistency

Disable support/counter role consistency.

Purpose:

```text
measure role confusion protection
```

### A8. Parallel Heads Only

Set:

```yaml
residual_rho: 0.0
```

Purpose:

```text
measure disconnected rationale risk
```

### A9. No Signed Evidence

Use single evidence mask.

Purpose:

```text
measure value of separate supporting/counter evidence masks
```

### A10. No Diversity Trace Selection

Use random sampling inside buckets.

Purpose:

```text
measure evidence diversity value
```

### A11. Single Detector Only vs Multi-Detector Adapters

Compare:

```text
BWGNN only
BWGNN + CARE-GNN
BWGNN + GCN/GAT
BWGNN + tree-neighbor baseline
```

Purpose:

```text
support detector-adaptable claim
```

## Optional Experimental Ablations

Only under `experimental/`:

```text
DHEF
CER regularizer
evidence-conflict bucket
adaptive lambda
risk-type prototype prompt
multi-sample LLM self-consistency
```

These must not be enabled in main configs.

## Implementation Files

Expected files:

```text
configs/experiments/ablation_*.yaml
src/gread_core/evaluation/ablation.py
scripts/run_ablations.sh
scripts/export_results.py
tests/unit/test_ablation_configs.py
```

## Required Tests

1. every ablation config loads.
2. every ablation config declares `paper_warning`.
3. main configs do not enable experimental features.
4. ablation configs do not silently change dataset split.
5. table exporter includes ablation names.

## Acceptance Criteria

This module is complete when:

* all required ablation configs exist;
* ablation runner can execute smoke ablations;
* output tables include main and ablation metrics;
* experimental features remain disabled by default.

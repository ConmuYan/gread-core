# Evaluation Protocol

## Purpose

Evaluation must measure detection quality, reasoning quality, faithfulness responsiveness, and non-redundancy beyond base score.

## Level 1: Detection Metrics

Required:

```text
ROC-AUC
AUPRC
F1
Precision@K
Recall@K
```

AUPRC and Recall@K are required for imbalanced fraud detection.

## Level 2: Reasoning Quality Metrics

Required:

```text
verifier_acceptance_rate
contract_violation_rate
risk_type_agreement
positive_evidence_f1
negative_evidence_f1
evidence_sparsity
template_validity_rate
```

If human annotation is available:

```text
human_risk_type_agreement
human_evidence_agreement
```

LLM-as-judge may be reported only as auxiliary analysis, not ground truth.

## Level 3: Tri-CEC

Counterfactual Evidence Consistency evaluates whether weakening supporting evidence reduces model confidence.

### CEC-score

```text
CEC_score =
mean[ sigmoid(final_logit(E)) - sigmoid(final_logit(E_minus)) > 0 ]
```

### CEC-type

```text
CEC_type =
mean[ p(risk_type | E) - p(risk_type | E_minus) > 0 ]
```

### CEC-evidence

```text
CEC_evidence =
mean[ p(supporting evidence | E) - p(supporting evidence | E_minus) > 0 ]
```

## Evidence Weakening

Allowed weakening operations:

```text
detector_signal high -> neutral
detector_signal_strength strong -> weak
neighbor_consistency low -> medium/high
feature_neighbor_discrepancy high -> medium/low
degree_level burst -> normal
```

Weakening must operate on MEP reasoning channel, not on `prediction_score`.

## Level 4: Non-Redundancy Test

Evaluate whether reasoning outputs add information beyond base score.

Models:

```text
Y ~ P
Y ~ P + T
Y ~ P + T + M
```

Where:

```text
Y = fraud label
P = base prediction score
T = risk type probabilities
M = signed evidence mask probabilities
```

Report:

```text
AUC and AUPRC for each model
delta_AUC
delta_AUPRC
```

If `P + T + M` does not improve over `P`, do not claim non-redundant reasoning.

## Implementation Files

Expected files:

```text
src/gread_core/evaluation/detection.py
src/gread_core/evaluation/reasoning.py
src/gread_core/evaluation/cec.py
src/gread_core/evaluation/non_redundancy.py
src/gread_core/evaluation/ablation.py
tests/unit/test_detection_metrics.py
tests/unit/test_reasoning_metrics.py
tests/integration/test_cec_pipeline.py
tests/unit/test_non_redundancy.py
```

## Required Tests

1. detection metrics match sklearn on fixture.
2. reasoning metrics handle empty accepted ERR set.
3. tri-CEC runs on tiny model.
4. evidence weakening does not modify prediction_score.
5. non-redundancy outputs all three model scores.
6. metric JSON files are saved.

## Acceptance Criteria

This module is complete when:

* all required metrics are implemented;
* evaluation writes JSON metrics;
* tri-CEC and non-redundancy run on smoke experiment;
* metric outputs are compatible with table exporter.

# Experimental Extensions

## Purpose

This spec records optional ideas from reviewer feedback and external critique without allowing them to silently enter the main method.

## Rule

All experimental extensions must be:

1. disabled by default;
2. implemented under `src/gread_core/experimental/`;
3. activated only by explicit config;
4. excluded from main paper claims unless separately ablated;
5. marked with `paper_warning`.

## Experimental Feature 1: DHEF

Dynamic Hard Evidence Filtering.

Purpose:

```text
filter accepted ERRs based on student prediction consistency
```

Status:

```text
experimental only
```

Main-method restriction:

```text
Do not multiply main reasoning loss by b_v unless running DHEF ablation.
```

Allowed config:

```yaml
experimental:
  dhef:
    enabled: false
```

## Experimental Feature 2: CER Regularizer

Counterfactual Evidence Regularization.

Purpose:

```text
turn CEC-style responsiveness into training regularizer
```

Status:

```text
experimental only
```

Main-method restriction:

```text
Main loss remains L_sup + lambda * a_v * (L_type + L_evidence).
```

Allowed config:

```yaml
experimental:
  cer_regularizer:
    enabled: false
```

## Experimental Feature 3: Evidence-Conflict Bucket

Purpose:

```text
sample nodes where prediction_score and detector evidence conflict
```

Status:

```text
experimental only
```

Main-method restriction:

```text
Main trace selection remains three-bucket + diversity.
```

Allowed config:

```yaml
experimental:
  evidence_conflict_bucket:
    enabled: false
```

## Experimental Feature 4: Adaptive Lambda

Purpose:

```text
schedule lambda based on verifier acceptance rate or model confidence
```

Status:

```text
experimental only
```

Main-method restriction:

```text
Default main method uses lambda_reason = 0.5.
```

Allowed config:

```yaml
experimental:
  adaptive_lambda:
    enabled: false
```

## Experimental Feature 5: Risk-Type Prototype Prompting

Purpose:

```text
inject risk-type prototypes into LLM teacher prompt
```

Status:

```text
experimental only
```

Main-method restriction:

```text
Default LLM prompt uses taxonomy and evidence rules only.
```

Allowed config:

```yaml
experimental:
  prototype_prompting:
    enabled: false
```

## Experimental Feature 6: Multi-Sample LLM Self-Consistency

Purpose:

```text
generate multiple ERRs per MEP and accept only stable outputs
```

Status:

```text
experimental only
```

Main-method restriction:

```text
Default teacher uses one deterministic generation with temperature 0.
```

Allowed config:

```yaml
experimental:
  llm_self_consistency:
    enabled: false
```

## Experimental Feature 7: Learned Verifier

Purpose:

```text
train a model to predict ERR quality
```

Status:

```text
not allowed in main method
```

Main-method restriction:

```text
The main verifier must remain deterministic.
```

## Implementation Files

Only if implemented:

```text
src/gread_core/experimental/dhef.py
src/gread_core/experimental/cer.py
src/gread_core/experimental/conflict_bucket.py
src/gread_core/experimental/adaptive_lambda.py
src/gread_core/experimental/prototype_prompting.py
src/gread_core/experimental/self_consistency.py
configs/experiments/experimental_*.yaml
```

## Required Tests

If any experimental feature is added:

1. default config keeps it disabled.
2. main experiment configs keep it disabled.
3. enabling it requires explicit experimental config.
4. output metrics include `experimental_feature_enabled`.
5. PR description marks it experimental.

## Acceptance Criteria

This spec is satisfied when:

* experimental ideas are documented;
* main method remains minimal;
* no optional feature silently changes main results;
* ablations can isolate optional feature effects.

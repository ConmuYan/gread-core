# Trace Node Selection

## Purpose

Trace selection chooses a small number of nodes for offline LLM ERR generation.

## Main Strategy

Use fixed three-bucket sampling with evidence diversity within each bucket.

Buckets:

```text
1. uncertain nodes
2. high-confidence fraud nodes
3. high-confidence benign nodes
```

Default budget split:

```yaml
uncertain: 0.333
high_conf_fraud: 0.333
high_conf_benign: 0.334
```

## Inputs

Trace selector may use:

```text
prediction_score
uncertainty
ground-truth labels for train nodes
MEP reasoning channel for diversity
```

Trace selector may not expose `prediction_score` to LLM prompts.

## Bucket Definitions

### Uncertain Nodes

Nodes with prediction score near decision boundary or high uncertainty.

Example:

```text
abs(prediction_score - 0.5) small
```

### High-Confidence Fraud Nodes

Training nodes with:

```text
label = fraud
prediction_score high
uncertainty low
```

### High-Confidence Benign Nodes

Training nodes with:

```text
label = benign
prediction_score low
uncertainty low
```

## Evidence Diversity Sampling

Within each bucket, select samples that maximize coverage over evidence patterns.

MEP reasoning channel is converted into a discrete vector:

```text
[
  uncertainty_level,
  degree_level,
  neighbor_consistency,
  feature_neighbor_discrepancy,
  detector_signal,
  detector_signal_strength,
  counter_signal
]
```

Sampling may use:

* greedy coverage
* deterministic hashing
* k-medoids
* clustering over evidence vectors

Must be deterministic under seed.

## Forbidden Main-Method Behavior

The following are not part of the main trace selector:

* evidence-conflict bucket
* dynamic budget transfer
* active learning loop
* DHEF-based sampling

These may exist only under `experimental/`.

## Implementation Files

Expected files:

```text
src/gread_core/tracing/buckets.py
src/gread_core/tracing/diversity.py
src/gread_core/tracing/selector.py
tests/unit/test_trace_selection.py
configs/default.yaml
```

## Required Tests

1. Three buckets are populated when eligible nodes exist.
2. Total selected nodes do not exceed budget.
3. Sampling is deterministic under seed.
4. Evidence diversity improves or equals random coverage on fixture.
5. `prediction_score` is used only for selection, not prompt payload.
6. Empty buckets are handled gracefully.

## Acceptance Criteria

This module is complete when:

* trace selection returns node IDs and selection metadata;
* selection metadata logs bucket source;
* diversity sampling is deterministic;
* no LLM call occurs;
* score leakage tests pass.

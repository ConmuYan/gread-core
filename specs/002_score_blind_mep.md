# Score-Blind Minimal Evidence Package

## Purpose

The Minimal Evidence Package provides compact detector-native evidence to the LLM teacher while preventing score leakage.

## Core Requirement

`prediction_score` must not influence LLM rationale generation.

## Data Structure

MEP has two channels:

```text
MinimalEvidencePackage
├── CalibrationChannel
└── ReasoningChannel
```

## CalibrationChannel

Used for trace selection, calibration analysis, and detection analysis.

Fields:

```yaml
prediction_score: float in [0, 1]
uncertainty: float in [0, 1]
```

Allowed consumers:

* trace selector
* detection evaluator
* experiment logger
* calibration analysis

Forbidden consumers:

* LLM prompt builder
* ERR generator payload
* evidence targets
* risk type targets
* reasoning loss

## ReasoningChannel

Used by LLM teacher and student evidence encoder.

Required fields:

```yaml
uncertainty_level: low | medium | high
degree_level: very_low | low | normal | high | burst | unavailable
neighbor_consistency: low | medium | high | unavailable
feature_neighbor_discrepancy: low | medium | high | unavailable
detector_signal: string
detector_signal_strength: weak | moderate | strong | unavailable
counter_signal: string
allowed_support_ids: list[string]
allowed_counter_ids: list[string]
```

## Teacher Payload

The only method allowed to produce LLM input is:

```python
MinimalEvidencePackage.to_teacher_payload()
```

It must return only:

```json
{
  "node_id": "...",
  "detector_name": "...",
  "reasoning": {
    ...
  }
}
```

It must exclude:

```text
calibration
prediction_score
raw fraud probability
base detector score
ground-truth label
node id of neighbors
edge ids
full graph topology
```

## Evidence Role Rules

`allowed_support_ids` may include:

```text
degree_level
neighbor_consistency
feature_neighbor_discrepancy
detector_signal
detector_signal_strength
```

`allowed_counter_ids` may include:

```text
counter_signal
uncertainty_level
benign_neighbor_signal
```

Forbidden:

```text
prediction_score in allowed_support_ids
prediction_score in allowed_counter_ids
counter_signal in allowed_support_ids
uncertainty_level as sole support for strong fraud type
```

## Implementation Files

Expected files:

```text
src/gread_core/schemas/evidence.py
src/gread_core/evidence/mep.py
src/gread_core/evidence/leakage_guard.py
tests/unit/test_mep_score_blind.py
tests/paper_alignment/test_prediction_score_not_in_prompt.py
scripts/check_no_leakage.py
```

## Unit Tests

Required tests:

1. `prediction_score` exists in `CalibrationChannel`.
2. `prediction_score` is absent from `to_teacher_payload()`.
3. `prediction_score` cannot be in `allowed_support_ids`.
4. `prediction_score` cannot be in `allowed_counter_ids`.
5. `counter_signal` cannot be in `allowed_support_ids`.
6. teacher payload serializes to JSON.
7. teacher payload contains only reasoning channel.

## Paper-Alignment Tests

Required checks:

```bash
python scripts/check_no_leakage.py
pytest tests/paper_alignment/test_prediction_score_not_in_prompt.py
```

## Acceptance Criteria

This module is complete when:

* MEP schema validates all required fields;
* teacher payload is score-blind by construction;
* leakage guard fails if prompt templates mention `prediction_score`;
* tests pass;
* no LLM code is implemented in this module.

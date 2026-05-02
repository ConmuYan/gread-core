# Evidence Rationale Record and Offline LLM Teacher

## Purpose

The LLM teacher generates structured Evidence Rationale Records from score-blind MEP teacher payloads.

LLM is used only offline during Stage 2.

## ERR Schema

An Evidence Rationale Record contains:

```json
{
  "risk_type": "spectral_anomaly",
  "supporting_evidence": ["detector_signal", "neighbor_consistency"],
  "counter_evidence": ["counter_signal"],
  "summary": "..."
}
```

## Risk Type Taxonomy

Allowed risk types:

```text
structural_discrepancy
camouflage_neighbor
spectral_anomaly
feature_structure_conflict
relation_or_burst_anomaly
weak_or_uncertain_evidence
```

## Training Targets

Only the following ERR fields may be used for training:

```text
risk_type
supporting_evidence
counter_evidence
```

Forbidden as training signal:

```text
summary
raw LLM text
chain of thought
LLM confidence
```

## Teacher Payload

LLM teacher must receive only:

```python
mep.to_teacher_payload()
```

It must not receive:

```text
prediction_score
ground-truth label
full graph topology
neighbor IDs
edge IDs
raw node identifiers beyond opaque node_id
```

## Prompt Rules

Prompt must instruct the LLM:

1. output valid JSON only;
2. choose exactly one risk type;
3. cite only allowed evidence IDs;
4. use `counter_signal` only as counter evidence;
5. never mention fraud score, probability, or prediction score;
6. not infer missing evidence;
7. use `weak_or_uncertain_evidence` when evidence is insufficient.

## LLM Cache

Every prompt must be cached by hash.

Cache record must include:

```json
{
  "prompt_hash": "...",
  "teacher_payload_hash": "...",
  "model_name": "...",
  "temperature": 0.0,
  "raw_response": "...",
  "parsed_err": {...},
  "verification_result": {...},
  "created_at": "..."
}
```

Replay mode must reproduce parsed ERRs without network calls.

## LLM Client Interface

Expected abstraction:

```python
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...
```

Implementation must support:

* live mode
* replay mode
* dry-run fixture mode

## Implementation Files

Expected files:

```text
src/gread_core/schemas/err.py
src/gread_core/llm/teacher.py
src/gread_core/llm/clients.py
src/gread_core/llm/prompt_builder.py
src/gread_core/llm/cache.py
src/gread_core/llm/templates/err_generation.j2
tests/unit/test_err_schema.py
tests/integration/test_llm_cache_replay.py
tests/paper_alignment/test_prediction_score_not_in_prompt.py
```

## Required Tests

1. ERR schema accepts valid records.
2. ERR schema rejects unknown risk types.
3. ERR schema rejects invalid evidence list types.
4. `summary` is absent from `training_targets()`.
5. prompt builder does not include `prediction_score`.
6. replay mode does not call network.
7. cache key is deterministic.
8. malformed LLM JSON is rejected or retried.

## Acceptance Criteria

This module is complete when:

* ERR schema is implemented;
* prompt is score-blind;
* LLM output is cached;
* replay mode works offline;
* parsed ERRs must pass Evidence Contract Verifier before becoming training targets;
* no inference module imports LLM code.

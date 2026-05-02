# LLM-Free Inference Protocol

## Purpose

Inference must produce fraud reasoning outputs without calling or importing LLM code.

## Inference Inputs

```text
Graph G
Features X
Target node IDs
trained base detector
trained GReaD reasoner
detector adapter
template explanation generator
```

## Inference Outputs

Each node output must include:

```json
{
  "node_id": "...",
  "fraud_score": 0.84,
  "risk_type": "spectral_anomaly",
  "supporting_evidence": ["detector_signal", "neighbor_consistency"],
  "counter_evidence": ["counter_signal"],
  "explanation": "..."
}
```

## Explanation Generation

Explanations must be deterministic templates.

Allowed inputs:

```text
predicted risk type
predicted positive evidence mask
predicted negative evidence mask
MEP reasoning values
```

Forbidden inputs:

```text
LLM output
ERR summary
chain-of-thought
network calls
prediction_score as evidence
```

## Inference Pipeline

```text
target node
  ↓
base detector forward
  ↓
adapter extracts MEP
  ↓
evidence encoder encodes reasoning channel
  ↓
GReaD reasoner outputs score/type/masks
  ↓
template generator creates explanation
```

## No-LLM Guard

Inference code must not import:

```text
gread_core.llm
openai
anthropic
requests
httpx
```

## Implementation Files

Expected files:

```text
src/gread_core/inference/predictor.py
src/gread_core/inference/explanation_template.py
src/gread_core/inference/no_llm_guard.py
tests/unit/test_explanation_template.py
tests/unit/test_no_llm_inference.py
tests/paper_alignment/test_inference_is_llm_free.py
scripts/check_no_llm_inference.py
```

## Required Tests

1. inference returns all required fields.
2. explanation is deterministic.
3. explanation contains risk type and evidence names.
4. inference imports no LLM code.
5. inference performs no network call.
6. `prediction_score` is not listed as supporting evidence.
7. batch inference works.

## Acceptance Criteria

This module is complete when:

* inference works from checkpoint and config;
* no LLM/network imports exist;
* explanations are template-based;
* output schema is stable;
* paper-alignment tests pass.

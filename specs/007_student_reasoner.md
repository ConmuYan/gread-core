# Evidence-Conditioned Student Reasoner

## Purpose

The student reasoner predicts fraud score, risk type, and signed evidence masks without online LLM calls.

## Inputs

The reasoner receives:

```text
base detector embedding z_v
base detector logit
MEP reasoning channel encoded as evidence tokens
```

It must not receive:

```text
LLM summary
raw LLM text
prediction_score as evidence token
online LLM output
```

## Outputs

Required outputs:

```python
{
  "base_logit": Tensor[B],
  "final_logit": Tensor[B],
  "type_logits": Tensor[B, T],
  "pos_mask_logits": Tensor[B, K],
  "neg_mask_logits": Tensor[B, K]
}
```

Where:

```text
T = number of risk types
K = number of evidence slots
```

## Architecture

### Evidence Encoder

Encodes MEP reasoning channel into:

```text
g_v = phi(E_v)
```

### Type Head

```text
type_logits = h_type([z_v; g_v])
```

### Signed Evidence Heads

```text
pos_mask_logits = h_pos([z_v; g_v])
neg_mask_logits = h_neg([z_v; g_v])
```

### Evidence-Gated Residual Readout

Base score remains dominant.

```text
final_logit = base_logit + rho * residual_logit
```

Default:

```yaml
rho: 0.1
```

Ablation:

```yaml
rho: 0.0
```

must recover base detector logits.

## Tensor Shapes

```text
z_v:                    [B, H]
evidence_token_ids:      [B, K]
evidence_embedding g_v:  [B, E]
base_logit:              [B]
final_logit:             [B]
type_logits:             [B, T]
pos_mask_logits:         [B, K]
neg_mask_logits:         [B, K]
```

## Forbidden Behavior

Model code must not import:

```text
gread_core.llm
openai
anthropic
requests
httpx
```

Reasoner must not use:

```text
ERR summary
prediction_score evidence token
LLM hidden chain-of-thought
```

## Implementation Files

Expected files:

```text
src/gread_core/models/evidence_encoder.py
src/gread_core/models/heads.py
src/gread_core/models/residual_readout.py
src/gread_core/models/reasoner.py
tests/unit/test_reasoner_shapes.py
tests/unit/test_residual_readout.py
tests/paper_alignment/test_signed_evidence_masks.py
tests/paper_alignment/test_inference_is_llm_free.py
```

## Required Tests

1. forward returns all required outputs.
2. output shapes match spec.
3. `rho=0` gives `final_logit == base_logit`.
4. evidence tokens affect type/evidence outputs.
5. signed evidence masks are separate.
6. no LLM imports in model files.
7. no prediction_score token is required by evidence encoder.

## Acceptance Criteria

This module is complete when:

* reasoner forward pass works on tiny batch;
* gradients flow through evidence heads;
* residual readout is config-controlled;
* LLM-free import guard passes;
* reasoner does not use ERR summary.

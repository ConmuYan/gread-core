# ADR_0003: LLM-Free Inference

## Status
Accepted

## Context
How do we ensure production inference does not require LLM dependencies?

## Decision
Separate training and inference paths. LLMs are only used during training (as teachers) and for data preparation. Inference uses distilled student models.

### Architecture
```
Training: LLM Teacher → Student Training → Student Model
Inference: Input → Student Model → Output
```

### Implementation Requirements
1. **Student Model**: Must be standalone (no LLM API calls)
2. **Model Export**: Student model saved in standard format
3. **Inference Path**: Explicit inference path that tests without LLM access

### Validation
```python
def test_inference_is_llm_free():
    # Mock LLM APIs to ensure they're not called
    with mock_llm_apis(fail_if_called=True):
        result = student_model.infer(input)
    return result
```

## Consequences
- **Positive**: Production deployment without LLM costs/latency
- **Positive**: Offline operation capability
- **Negative**: Distillation overhead during training
- **Mitigation**: One-time distillation per domain

## Alternatives Considered
1. **LLM at inference**: Call LLM for each query - Rejected due to cost/latency
2. **Hybrid**: Cache LLM responses - Rejected due to staleness concerns

---

*Proposed: 2026-04-30 | Accepted: 2026-04-30*

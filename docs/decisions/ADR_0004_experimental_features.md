# ADR_0004: Experimental Features

## Status
Accepted

## Context
How do we handle experimental features that may not be production-ready?

## Decision
Experimental features are gated behind feature flags and clearly marked.

### Feature Categories
1. **Stable**: Production-ready, fully verified
2. **Experimental**: Under development, may change
3. **Deprecated**: Will be removed

### Implementation
```python
# Experimental feature marker
from gread_core.experimental import feature

@feature(status="experimental", id="mep_v2")
def mep_v2(G, epsilon):
    """Next-gen MEP - may change without notice."""
    ...
```

### Documentation Requirements
- Experimental features must be marked in docs
- Migration guide for features changing status
- Deprecation timeline for deprecated features

## Consequences
- **Positive**: Users can opt-in to bleeding-edge features
- **Positive**: Clear expectations about stability
- **Negative**: API surface is larger
- **Mitigation**: Clear status indicators in docs

## Alternatives Considered
1. **Separate branch**: Experimental features in dev branch - Rejected due to merge overhead
2. **No experiments**: Only release stable features - Rejected due to slower feedback loop

---

*Proposed: 2026-04-30 | Accepted: 2026-04-30*

# ADR_0002: Evidence Contract Verifier

## Status
Accepted

## Context
How do we ensure all claims in the codebase are empirically verified?

## Decision
Implement an Evidence Contract Verifier that maps each claim to specific empirical evidence.

### Contract Structure
```python
class Claim:
    description: str
    evidence: List[Evidence]
    status: Literal["verified", "pending", "rejected"]

class Evidence:
    type: Literal["test", "experiment", "proof"]
    location: str  # file/line or experiment ID
    metric: Optional[str]
```

### Verification Protocol
1. Each claim must map to ≥1 evidence
2. Evidence must be reproducible
3. Verification is part of completion criteria

## Consequences
- **Positive**: All claims have backing evidence; easier to defend
- **Positive**: Test coverage is meaningfully measured
- **Negative**: Additional overhead for claim documentation
- **Mitigation**: Automated claim extraction from code comments/docs

## Alternatives Considered
1. **Standard testing**: Just write tests - Rejected due to lack of claim-evidence mapping
2. **Documentation only**: Trust that docs match code - Rejected as unverifiable

---

*Proposed: 2026-04-30 | Accepted: 2026-04-30*

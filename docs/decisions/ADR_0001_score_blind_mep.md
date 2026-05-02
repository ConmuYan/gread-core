# ADR_0001: Score-Blind MEP

## Status
Accepted

## Context
How do we perform Maximum Entropy Perturbation without access to ground-truth anomaly labels?

## Decision
Use entropy-maximizing perturbations that are agnostic to ground-truth labels. The perturbation strategy maximizes the entropy of the resulting graph distribution without any label information.

### Key Points
1. **No Label Access**: Perturbation module never sees ground-truth scores
2. **Entropy Objective**: Maximize H(G') where G' is perturbed graph
3. **Structure-Aware**: Preserve global graph properties while perturbing

### Implementation
```python
def entropy_maximizing_perturbation(G, epsilon):
    # No labels used here
    # Maximize entropy of edge/feature changes
    return G_perturbed
```

## Consequences
- **Positive**: True unsupervised operation; applicable where labels unavailable
- **Negative**: May require more iterations to converge
- **Mitigation**: Use domain-specific entropy approximations

## Alternatives Considered
1. **Semi-supervised MEP**: Use limited labels - Rejected due to score-blind requirement
2. **Heuristic perturbation**: Random/structural perturbations - Rejected due to lack of theoretical guarantee

---

*Proposed: 2026-04-30 | Accepted: 2026-04-30*

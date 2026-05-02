# Experiment Lifecycle

## Standard Workflow

### 1. Design Phase
- Define hypothesis
- Specify metrics
- Plan ablations
- Document in `specs/` or `docs/research/`

### 2. Implementation Phase
- Implement experiment
- Add logging/tracking
- Create reproducibility artifacts

### 3. Execution Phase
- Run with seed control
- Monitor for anomalies
- Record all outputs

### 4. Analysis Phase
- Collect metrics
- Compare to baselines
- Generate visualizations

### 5. Verification Phase
- Run evidence verifier
- Document claims
- Archive results

## Reproducibility Requirements

### Seed Control
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
```

### Environment Tracking
- Python version
- Package versions
- GPU/CUDA info
- Commit hash

### Result Archival
- Raw outputs
- Metrics logs
- Visualization artifacts
- Interpretation notes

## Claim Documentation

### Claim Format
```markdown
## Claim: [Description]

**Evidence**: [What supports this]
**Metric**: [Quantitative measure]
**Baseline**: [Comparison point]
**Status**: Verified | Pending | Rejected
```

## Experiment Checklist

- [ ] Hypothesis clearly stated
- [ ] Metrics predefined
- [ ] Seeds controlled
- [ ] Environment documented
- [ ] Results archived
- [ ] Claims verified

---

*See `specs/007_evaluation_protocol.md` for evaluation details*

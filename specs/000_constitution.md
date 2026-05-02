# GReaD-Core Constitution

## Project Identity

GReaD-Core implements:

**Contract-Verified Score-Blind Evidence Distillation for LLM-Free Graph Fraud Reasoning.**

The framework converts a trained graph fraud detector into an evidence-grounded reasoner by using offline LLM-generated Evidence Rationale Records, accepted only through deterministic Evidence Contract Verification.

## Non-Negotiable Principles

### P1. Paper-first implementation

Every module must map to one of the following paper components:

1. Score-blind Minimal Evidence Package
2. Detector-Evidence Adapter Protocol
3. Evidence Rationale Record
4. Evidence Contract Verifier
5. Offline LLM teacher
6. Evidence-conditioned residual student reasoner
7. Three-stage training
8. LLM-free inference
9. Tri-CEC and non-redundancy evaluation

If a change does not map to one of these components, it must be placed under `experimental/` and disabled by default.

### P2. Score-blind reasoning

`prediction_score` is calibration-only.

It may appear only in:

- `CalibrationChannel`
- trace selection
- calibration analysis
- detection metric analysis

It must never appear in:

- LLM teacher prompt
- `supporting_evidence`
- `counter_evidence`
- evidence mask labels
- risk type generation payload
- training targets for reasoning heads

### P3. Offline-only LLM

LLM usage is allowed only in Stage 2:

```text
MEP -> prompt -> LLM teacher -> ERR -> Evidence Contract Verifier -> cache
```

LLM usage is forbidden in:

* model forward pass
* training Stage 1
* training Stage 3
* inference
* evaluation, unless explicitly evaluating teacher quality offline

The inference path must not import:

```text
gread_core.llm
openai
anthropic
requests
httpx
```

### P4. Deterministic verifier

The main verifier must be deterministic and rule-based.

Forbidden in the main verifier:

* LLM-as-judge
* learned verifier
* soft confidence score as acceptance
* stochastic verification

Accepted ERR requires all checks:

1. schema validity
2. evidence availability
3. role consistency
4. risk-evidence contract consistency
5. score-blindness
6. label compatibility

### P5. Accepted ERR only

The reasoning loss is applied only when `accepted = 1`.

Main objective:

```text
L = L_sup + lambda * a_v * (L_type + L_evidence)
```

Rejected ERRs must contribute zero type/evidence loss.

### P6. Free-form text is not a label

ERR `summary` may be used only for:

* logging
* human inspection
* qualitative examples

ERR `summary` must not be used as:

* training label
* embedding input
* target for evidence head
* input to inference model

### P7. Main method must remain minimal

The main method includes:

* score-blind MEP
* detector adapter protocol
* evidence diversity trace selection
* Evidence Contract Verifier
* evidence-conditioned residual reasoner
* signed evidence masks
* LLM-free inference
* tri-CEC evaluation
* non-redundancy evaluation

The following are experimental only and disabled by default:

* DHEF
* CER as training regularizer
* evidence-conflict bucket
* adaptive lambda
* risk-type prototype prompting
* multi-sample LLM self-consistency
* learned verifier

### P8. Reproducibility

Every experiment must log:

* seed
* config hash
* git commit
* dataset split hash
* detector checkpoint path
* ERR cache hash
* contract version
* timestamp
* software version

### P9. Required checks

Every implementation task must pass:

```bash
ruff check .
mypy src
pytest tests/unit
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py
```

Model or training changes must also pass:

```bash
bash scripts/run_smoke.sh
```

## Definition of Done

A module is complete only when:

1. implementation matches the corresponding spec;
2. unit tests cover positive and negative cases;
3. paper-alignment tests pass;
4. no score leakage exists;
5. inference remains LLM-free;
6. docs and configs are updated;
7. `PROJECT_STATE.md` is updated.

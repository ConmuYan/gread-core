# Training Protocol

## Purpose

Training must avoid chicken-and-egg collapse by separating base detector warm-up, offline ERR generation, and reasoner distillation.

## Stages

### Stage 1: Base Detector Warm-up

Train base detector using supervised fraud labels only.

Objective:

```text
L_sup
```

Allowed outputs:

```text
base detector checkpoint
base logits
node embeddings
uncertainty estimates
MEP extraction cache
```

Forbidden:

```text
LLM calls
ERR generation
reasoning loss
risk type head training
```

### Stage 2: Offline ERR Generation and Verification

Inputs:

```text
trained base detector
trace nodes
score-blind MEPs
LLM teacher
Evidence Contract Verifier
```

Outputs:

```text
ERR cache
verification logs
accepted ERR training targets
rejected ERR logs
```

Only Stage 2 may call LLM.

### Stage 3: Student Reasoner Distillation

Train student reasoner with:

```text
L = L_sup + lambda * a_v * (L_type + L_evidence)
```

Default:

```yaml
lambda_reason: 0.5
```

Optional warm-up schedule may be used only if config explicitly enables it.

## Loss Definitions

### Supervised Loss

```text
L_sup = BCEWithLogits(final_logit, fraud_label)
```

### Type Loss

Applied only when `a_v=1`:

```text
L_type = CrossEntropy(type_logits, risk_type_target)
```

### Evidence Loss

Applied only when `a_v=1`:

```text
L_evidence =
  BCE(pos_mask_logits, pos_evidence_targets)
  + BCE(neg_mask_logits, neg_evidence_targets)
```

### Total Loss

```text
L = L_sup + lambda_reason * accepted_mask * (L_type + L_evidence)
```

## Rejected Samples

If `accepted_mask.sum() == 0`, then:

```text
type_loss = 0
evidence_loss = 0
total_loss = L_sup
```

## Forbidden Main-Method Training

Do not enable by default:

* DHEF
* CER regularizer
* adaptive lambda
* evidence-conflict sampling
* learned verifier feedback
* LLM-in-the-loop training

These belong to `experimental/`.

## Implementation Files

Expected files:

```text
src/gread_core/losses/supervised.py
src/gread_core/losses/reasoning.py
src/gread_core/training/stage1_train_detector.py
src/gread_core/training/stage2_generate_err.py
src/gread_core/training/stage3_train_reasoner.py
src/gread_core/training/trainer.py
src/gread_core/training/checkpointing.py
tests/unit/test_loss_masking.py
tests/integration/test_stage1_stage2_stage3_tiny.py
```

## Required Tests

1. Stage 1 does not import LLM.
2. Stage 2 is the only stage that can call LLM.
3. Stage 3 uses accepted ERR only.
4. rejected ERRs produce zero type/evidence loss.
5. `summary` is never used in loss.
6. tiny graph smoke training runs on CPU.
7. checkpoints include metadata.

## Acceptance Criteria

This module is complete when:

* all three stages run independently;
* Stage 2 cache can be replayed;
* Stage 3 trains with accepted ERR targets;
* smoke test passes;
* training metadata is saved.

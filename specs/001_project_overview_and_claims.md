# Project Overview and Claims

## Method Name

GReaD-Core

## Long Name

Contract-Verified Score-Blind Evidence Distillation for LLM-Free Graph Fraud Reasoning

## Core Pipeline

```text
Graph + Features
      ↓
Base Fraud Detector
      ↓
Detector-Evidence Adapter
      ↓
Score-Blind Minimal Evidence Package
      ↓
Offline LLM Teacher
      ↓
Evidence Rationale Record
      ↓
Evidence Contract Verifier
      ↓
Accepted ERR Cache
      ↓
Evidence-Conditioned Student Reasoner
      ↓
LLM-Free Fraud Reasoning
```

## Main Research Claim

GReaD-Core converts a trained graph fraud detector into an LLM-free evidence-grounded reasoner by distilling only contract-verified, score-blind, detector-native evidence rationales generated offline by an LLM teacher.

## Contributions

### C1. Score-Blind Detector-Native Evidence Interface

The method defines a score-blind Minimal Evidence Package that exposes detector-native evidence while preventing direct leakage from base prediction score to LLM rationale generation.

### C2. Contract-Verified Reasoning Distillation

The method accepts LLM-generated Evidence Rationale Records only when they satisfy deterministic Evidence Contract Verification.

### C3. Evidence-Conditioned LLM-Free Reasoner

The student model predicts fraud score, risk type, signed evidence masks, and deterministic template explanations without online LLM calls.

## Allowed Claims

The paper may claim:

* LLM-free inference
* score-blind reasoning prompt
* deterministic contract verification
* detector-native evidence distillation
* evidence-conditioned residual reasoning
* contract-consistent rationales
* counterfactual evidence responsiveness
* non-redundant reasoning outputs if supported by experiments

## Forbidden Claims

The paper must not claim:

* causal explanation guarantee
* universal any-detector compatibility
* LLM rationale is ground truth
* verifier proves semantic truth
* accepted ERRs are necessarily correct
* evidence masks are guaranteed causal explanations
* method works without detector-native evidence

## Correct Wording

Use:

```text
contract-consistent
score-blind
detector-adaptable
LLM-free inference
counterfactually responsive
```

Avoid:

```text
causally faithful
universally general
semantically proven
hallucination-free
guaranteed correct explanation
```

## Main Baselines

The implementation must support comparison with:

* base detector only
* base detector + naive heads
* base detector + LLM ERR without verifier
* base detector + schema-only verifier
* base detector + Evidence Contract Verifier
* detector-specific baselines
* tree ensemble + neighborhood aggregation when applicable

## Success Criteria

A successful implementation must show:

1. detection performance is not degraded;
2. reasoning outputs are valid and sparse;
3. accepted ERRs are contract-consistent;
4. tri-CEC scores improve over disconnected-head variants;
5. reasoning outputs provide non-redundant information beyond base score;
6. inference makes no LLM calls.

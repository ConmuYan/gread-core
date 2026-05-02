# AGENTS.md

## Project Identity

This repository implements GReaD-Core:
Contract-Verified Score-Blind Evidence Distillation for LLM-Free Graph Fraud Reasoning.

## Non-Negotiable Research Constraints

1. prediction_score is calibration-only.
   - It may appear in CalibrationChannel.
   - It must never appear in LLM prompts.
   - It must never appear in supporting_evidence or counter_evidence.
   - It must never be used as an evidence target.

2. LLM is training-offline only.
   - LLM code must stay under src/gread_core/llm.
   - Inference code must not import gread_core.llm.
   - Model code must not import OpenAI, Anthropic, requests, httpx, or other online LLM/network clients.

3. Evidence Contract Verifier must be deterministic.
   - No LLM-as-judge in the main verifier.
   - No learned verifier in the main method.
   - Accepted ERR requires schema, availability, role consistency, contract consistency, score-blindness, and label compatibility.

4. Training objective:
   L = L_sup + lambda * a_v * (L_type + L_evidence).
   - Rejected ERR samples must not contribute to type/evidence losses.
   - ERR summary must not be used for training.
   - DHEF, CER, ECB, adaptive lambda are experimental only and disabled by default.

5. Inference must output:
   - fraud_score
   - risk_type
   - supporting_evidence
   - counter_evidence
   - deterministic template explanation
   - no LLM call

## Required Validation Commands

Run before finishing any implementation task:

```bash
ruff check .
mypy src
pytest tests/unit
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py
````

For training/model changes, also run:

```bash
bash scripts/run_smoke.sh
```

## Main Method vs Experimental

Main method:

* score-blind MEP
* detector adapter protocol
* evidence diversity trace selection
* Evidence Contract Verifier
* evidence-conditioned residual reasoner
* signed evidence masks
* tri-CEC evaluation
* non-redundancy evaluation

Experimental only, disabled by default:

* DHEF
* CER as training regularizer
* evidence-conflict bucket
* multi-sample LLM self-consistency
* prototype prompt update
* adaptive lambda

````

Codex 官方说明，Codex 会在开始工作前读取 `AGENTS.md`，并按全局、项目、子目录的层级合并指令；靠近当前目录的文件在合并提示中靠后，因此能覆盖更上层的指导。:contentReference[oaicite:4]{index=4}

---


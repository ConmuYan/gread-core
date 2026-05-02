# GReaD-Core

Contract-Verified Score-Blind Evidence Distillation for LLM-Free Graph Fraud Reasoning.

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Smoke Test (CPU, ~30 seconds)

```bash
bash scripts/run_smoke.sh
```

Runs static analysis, unit tests, and the full 3-stage pipeline on a tiny synthetic graph.

### Main Experiment

```bash
# GCN on tiny graph (always available)
bash scripts/run_main_table.sh configs/experiments/main_gcn_tiny.yaml tiny gcn 1

# GCN on YelpChi (requires PyG FraudDataset)
bash scripts/run_main_table.sh configs/experiments/main_gcn_tiny.yaml yelpchi gcn 1
```

### Ablation Studies

```bash
# Run all ablation configs
bash scripts/run_ablations.sh tiny gcn 1

# Or run a specific ablation
bash scripts/run_main_table.sh configs/experiments/ablation_score_visible.yaml tiny gcn 1
```

### Export Paper Tables

```bash
python scripts/export_results.py
# Outputs: artifacts/tables/main_table.csv, artifacts/tables/ablation_table.csv
```

## Pipeline

GReaD-Core trains in 3 stages:

1. **Stage 1: Detector warm-up** — train a base GNN detector (GCN/GAT)
2. **Stage 2: ERR generation** — generate Evidence Rationale Records via LLM (offline, cache-replayable)
3. **Stage 3: Reasoner distillation** — train evidence-conditioned reasoner from accepted ERRs

Inference is fully LLM-free: the reasoner outputs `fraud_score`, `risk_type`, `supporting_evidence`, `counter_evidence`, and a deterministic template explanation.

## CLI Entry Points

```bash
python -m gread_core.cli.train_detector --config <config> --dataset <dataset> --detector gcn
python -m gread_core.cli.generate_err --config <config> --checkpoint <ckpt> --llm-backend stub
python -m gread_core.cli.train_reasoner --config <config> --detector-checkpoint <ckpt> --err-dir <errs>
python -m gread_core.cli.evaluate --checkpoint <ckpt> --config <config>
```

## Key Constraints

- `prediction_score` is calibration-only, never in prompts or evidence
- LLM is training-offline only; inference has zero LLM imports
- Evidence Contract Verifier is deterministic (no LLM-as-judge)
- Rejected ERRs never contribute to training loss

## Documentation

- [Implementation Blueprint](docs/engineering/IMPLEMENTATION_BLUEPRINT.md)
- [Experiment Lifecycle](docs/engineering/EXPERIMENT_LIFECYCLE.md)
- [Paper Claims and Non-Claims](docs/research/PAPER_CLAIMS_AND_NON_CLAIMS.md)
- [Architecture Decision Records](docs/decisions/)

## License

TBD

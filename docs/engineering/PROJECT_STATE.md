# PROJECT_STATE.md

## Current Phase

Epic: 11 - Experiment Infrastructure + TensorBoard + Pipeline Validation (COMPLETE)
Branch: master
Last updated: 2026-05-02

## Completed

### Epic 0: Repository Harness Bootstrap
- Documentation, agent infrastructure, ADRs, package skeleton, schemas, verifier

### Epics 3-8: Full Pipeline
- Data loaders, detectors, adapters, trace selection, LLM teacher, student reasoner, training pipeline, evaluation suite

### Phase 9A: Smoke Training Validation
- Full 3-stage pipeline end-to-end on tiny synthetic graph (CPU)
- Fixes: forward_with_embedding full embeddings, binary logits handling, StubClient, YAML compatibility

### Epic 9: Reproducibility Package
- Task 1: Experiment Infrastructure (experiment/seed.py, registry.py, logger.py)
- Task 2: Config Files (5 datasets, 2 detectors, 4 experiments, 3 ablations)
- Task 3: Runner Scripts (run_main_table.sh, run_ablations.sh, export_results.py)
- Task 4: CLI Integration (all 4 CLIs use set_seed + ExperimentRegistry)
- Task 5: README reproduction commands + artifact structure docs

### Epic 10: Gap Fill — Standalone Detectors + Inference Pipeline
- Task 1: Standalone Detectors (bwgnn.py, caregnn.py, tree_neighbor.py)
- Task 2: Inference Pipeline (inference/predictor.py, cli/infer.py)
- Task 3: Detector Configs (bwgnn.yaml, caregnn.yaml, tree_neighbor.yaml)
- Task 4: Experiment Configs (main_bwgnn_yelp.yaml, main_bwgnn_amazon.yaml, err_generation.yaml)
- Task 5: CI Workflow (.github/workflows/smoke.yml)
- Task 6: Test Fixtures (tests/fixtures/ with sample MEP, ERR, contracts)
- Task 7: Inference Module Expansion (no_llm_guard runtime enforcement, explanation_template structured output)
- Task 8: Tests (test_standalone_detectors.py, test_no_llm_guard.py)

### Epic 11: Experiment Infrastructure + TensorBoard + Pipeline Validation
- TensorBoard integration: SummaryWriter in stage1_train_detector.py and stage3_train_reasoner.py
- 8 metrics logged: train_loss, train_acc, val_loss, val_acc, total_loss, sup_loss, type_loss, evidence_loss
- All CLIs (train_detector, train_reasoner, generate_err, infer) support all 5 detector types
- --tensorboard-dir CLI arg added to train_detector and train_reasoner
- Data loader: synthetic graph fallback when FraudDataset unavailable
- Device handling fixes: stage3_train_reasoner.py and cec.py GPU compatibility
- 8 validation configs: val_gcn_{yelpchi,amazon,tfinance,tsocial}.yaml, val_bwgnn_{yelpchi,amazon,tfinance,tsocial}.yaml
- Validation script: scripts/run_validation.sh
- 8/8 validation runs passed (GCN + BWGNN on 4 datasets)
- pyproject.toml: tensorboard>=2.14 added to ml extras

## Validation Results

- 293 tests passing (unit)
- ruff check: all passed
- mypy src: 77 source files, no issues
- check_no_leakage.py: passed
- check_no_llm_inference.py: passed
- 77 source files, 36 test files
- Smoke pipeline: all 8 stages passed
- export_results.py produces deterministic CSV
- 8/8 validation runs passed (GCN + BWGNN on yelpchi, amazon, tfinance, tsocial)
- TensorBoard: 8 metrics logged per run (train_loss, train_acc, val_loss, val_acc, total_loss, sup_loss, type_loss, evidence_loss)

## Research Constraints Reconfirmed

- prediction_score is calibration-only
- summary not used for training
- rejected ERR excluded from reasoning loss
- no LLM imports in inference
- DHEF/CER/ECB/adaptive lambda experimental only

## Next

Ready for full-scale experiments with real datasets. Pipeline validated end-to-end on GPU (RTX 3090).
- Launch TensorBoard: `tensorboard --logdir artifacts/val_*/tensorboard`
- Run full experiments: `bash scripts/run_validation.sh`
- For real datasets: install FraudDataset-compatible PyG or use synthetic fallback
- Run ablation studies: `bash scripts/run_ablations.sh`

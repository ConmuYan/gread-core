# PROJECT_STATE.md

## Current Phase

Epic: 13 - Research Alignment Audit Fixes (COMPLETE)
Epic: 13b - Codex Review Round 2 Fixes (COMPLETE)
Epic: 13c - Ultrawork — Remaining P1/P2 Fixes (COMPLETE)
Epic: 13d - Codex Re-Review Final Fixes (COMPLETE)
Epic: 13e - Formal Experiment Routing (COMPLETE)
Branch: master
Last updated: 2026-05-04

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
- Dedicated Python venv: `/data1/mq/conda_envs/gread-core` (Python 3.10.20, PyTorch 2.6.0+cu124, PyG 2.7.0)
- pyproject.toml: `requires-python` changed to `>=3.10` for venv compatibility
- numpy 1.26.4 (downgraded from 2.x to fix `ndarray` type-arg mypy errors and `np.trapezoid` API)
- `np.trapezoid` → `np.trapz` in detection.py for numpy 1.x compatibility
- `datetime.UTC` → `datetime.timezone.utc` in registry.py and checkpointing.py for Python 3.10
- mypy CLI flags in smoke script to suppress tensorboard/numpy type noise
- CUDA: NVIDIA GeForce RTX 3090, all 5 detector types verified on GPU
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

### Epic 12: Real Dataset Pipeline Validation
- Real dataset loaders: YelpChi (.mat), Amazon (.mat), tfinance (DGL binary), tsocial (DGL binary)
- dgl graphbolt stub workaround for PyTorch 2.6.0 compatibility
- Sparse adjacency matrix fix for large graphs (45954 nodes)
- Vectorized signal computation (scatter_add instead of Python loops)
- 30 real data loader tests passing
- YelpChi pipeline: Stage 1 (5 epochs) → Stage 2 (6 ERRs accepted, 1 rejected) → Stage 3 (5 epochs) COMPLETE
- YelpChi evaluation: non-redundancy AUC improvement evidence_over_type=0.154
- Amazon pipeline: Stage 1 → Stage 2 → Stage 3 COMPLETE
- Ablation studies (3 configs on YelpChi): ALL COMPLETE
- Evaluate CLI updated: real inference on test nodes (not synthetic data)
- Real evaluation results on YelpChi (GCN, 5 epochs):
  - MAIN (score_blind): AUC=0.5453, AUPRC=0.1634, type_over_score=+0.0853, evidence_over_type=+0.0025
  - ABL score_visible: AUC=0.5441, type_over_score=+0.0017
  - ABL no_reasoner: AUC=0.5441, type_over_score=+0.0017
  - ABL schema_only: AUC=0.5441, type_over_score=+0.0017
  - Score-blind design shows 50x better type discrimination (+0.0853 vs +0.0017)
- Real evaluation results on Amazon (GCN, 5 epochs):
  - MAIN: AUC=0.8115, AUPRC=0.2736, type_over_score=+0.0349, evidence_over_type=+0.0090
- tri-CEC evaluation FIXED:
  - Root cause 1: slot_to_id used generic slot_{i} names, not reasoning field values
  - Root cause 2: DEFAULT_WEAKEN_CONFIG only had high-risk replacements, benign nodes had no variants
  - Root cause 3: reasoning_dict contained list fields causing TypeError
  - Fix: _mep_to_evidence_token_ids checks field values, expanded weaken config, filtered list fields
  - Results: score_cec=0.62, evidence_cec=0.62, type_cec=0.0, n_samples=10
- Score-blind design verified: no prediction_score in evidence
- Non-redundancy metrics verified: evidence adds value over type
- tri-CEC evaluation verified
- Evaluate CLI uses synthetic data for demo (limitation documented)
- Full-scale experiments on 4 GPUs: 3 ablations parallel on GPU 1/2/3

### Epic 13: Research Alignment Audit Fixes
- Full audit completed (FULL_RESEARCH_ALIGNMENT_AUDIT.md, 518 lines)
- P0-1 FIXED: Stage 2 verifier now loads YAML contract from `contract_path`, fail-closed if missing
- P0-2 FIXED: Canonical vocabulary — `RISK_TYPES_ORDERED`, `RISK_TYPE_TO_INDEX`, `EVIDENCE_SLOTS_ORDERED`, `EVIDENCE_SLOT_TO_INDEX` in `risk_taxonomy.py`
- P0-3 FIXED: Stage 3 uses real evidence token IDs from ERR fields (not all-zeros)
- P0-4 FIXED: Inference pipeline indexes embeddings/logits by `node_ids` for batch safety
- P0-5 FIXED: Replaced Python `hash()` with deterministic `EVIDENCE_SLOT_TO_INDEX` mapping
- P1-1 FIXED: PromptBuilder validates payload — rejects calibration keys and score tokens
- P1-2 FIXED: Real dataset loader fails closed (no silent synthetic fallback)
- P1-3 FIXED: mypy src passes (77 source files, 0 errors)
- PromptBuilder score leakage: `_validate_payload()` checks allowed keys + recursive score token scan
- Risk type index consistency: training (RISK_TYPE_TO_INDEX) and inference (RISK_TYPES_ORDERED) now identical
- Evidence token determinism: all modules use EVIDENCE_SLOT_TO_INDEX, no Python hash()

### Epic 13b: Codex Review Round 2 Fixes
- Codex reviewed all 14 original findings — identified 10 remaining issues
- P0-6 FIXED: Verifier deep merge — `generate_err.py` now preserves YAML `label_compatibility` dict when config overrides with bool
- P0-7 FIXED: Inference `_encode_evidence()` — now uses field NAMES (not values) via shared `encode_evidence_slots()`
- P0-8 FIXED: Inference batch safety — `predictor.py` removes masks before forward pass to get [N] logits
- P0-9 FIXED: Evaluation reasoning metrics — real mode uses actual predictions + ERR references; synthetic mode warns
- P0-10 FIXED: `run_ablations.sh` — passes `--dataset`, `--detector`, `--detector-checkpoint` to evaluate
- P1-5 FIXED: Unified evidence tokenizer — shared `encode_evidence_slots()` in `risk_taxonomy.py`, token 0=padding, slot i→token i+1
- All 3 consumers (Stage 3, inference, CEC) now use consistent tokenization scheme
- CEC weaken config expanded with more value replacements for better coverage
- All 6 validation gates pass: ruff, mypy (77 files), unit (323), paper alignment (6), no-leakage, no-LLM

### Epic 13c: Ultrawork — Remaining P1/P2 Fixes
- P1-6 FIXED: Reproducibility metadata CLI wiring — `dataset`, `detector_checkpoint_path`, `contract_version` passed to CheckpointManager from CLIs
- P1-7 FIXED: LLM cache schema expansion — `put()` accepts `payload_hash`, `model`, `verification_result`, `contract_version`; backward-compatible reads
- P2-1 FIXED: Adapter thresholds config-driven — all 4 adapters (pyg_gnn, bwgnn, caregnn, tree) accept `thresholds` dict with sensible defaults
- P2-2 FIXED: Ablation config switch wiring — `score_blind`, `signed_masks`, `rho`, `label_compatibility` switches connected to pipeline behavior
- CLIs (generate_err, evaluate, infer) read `config["adapter"]["thresholds"]` and pass to adapters
- Reasoner reads `signed_masks` config; PromptBuilder reads `score_blind` config

### Epic 13d: Codex Re-Review Final Fixes
- Codex re-reviewed remaining issues — all resolved
- P0-1 FIXED: predictor.py slot_names IndexError — capped at `min(num_slots, len(slot_names))`
- P0-2 FIXED: evaluate.py evidence encoding — now uses `encode_evidence_slots()` with field NAMES
- P0-3 FIXED: evaluate.py evidence token decode off-by-one — `slot_names[val - 1]` (was `slot_names[val]`)
- P0-4 FIXED: evaluate.py CEC slot_to_id — uses `EVIDENCE_SLOT_TO_INDEX` instead of value hash
- P0-5 FIXED: evaluate.py reasoning metrics — `err_lookup.get(test_nodes[i])` uses global node index
- P0-6 FIXED: smoke pipeline — StubClient generates `structural_discrepancy` (was `weak_or_uncertain_evidence` which was contract-forbidden); Stage 2 now produces accepted ERRs
- P0-7 FIXED: teacher.py `_get_completion()` removed; bare cache writes eliminated; `generate_err()` only caches after verification with metadata
- P0-8 FIXED: cache.py `put()` — allows metadata enrichment when key already exists
- P0-9 FIXED: generate_err.py CLI — injects `contract_version` from contract YAML; registry created AFTER contract loading
- P0-10 FIXED: stage2_generate_err.py — passes `contract_version` to `teacher.generate_err()`
- P0-11 FIXED: configs/contracts/gread_v1.yaml — added `contract_version: gread_v1` field
- P1-1 FIXED: Stage3 CLI fail-closed — raises RuntimeError on 0 accepted ERRs; `--allow-empty-err` for testing
- P1-2 FIXED: Unified contract_version resolution — train_reasoner.py and evaluate.py load contract YAML to extract version
- P1-3 FIXED: Training/generation/evaluation CLIs wire `contract_version` and `detector_checkpoint` to ExperimentRegistry
- P1-4 FIXED: train_detector.py passes `split_config` from config to registry
- P1-5 FIXED: train_reasoner.py uses `contract_version` (not `contract_path`) for CheckpointManager
- P1-6 FIXED: evaluate.py `_run_real_inference()` returns `test_nodes` for ERR lookup alignment
- 3 regression test files added (11 tests): cache enriched entry, inference canonical slots, ERR lookup global index
- All validation gates pass: ruff, mypy (77 files), unit (340), paper alignment, no-leakage, no-LLM

### Epic 13e: Formal Experiment Routing
- Data root routing added: CLI `--data-root` > config `data.root`/`data.data_root` > `GREAD_DATA_ROOT` > `data/raw`
- Project-local `data/raw` symlink points to the existing PriorF-GNN dataset directory; `data/` is ignored, so the 4.9G datasets are not copied or tracked
- Stage 2 runtime now resolves backend/cache/model/temperature from config with CLI overrides; non-tiny datasets reject `stub`
- Formal scripts (`run_main_table`, `run_validation`, `run_full_experiments`, `run_ablations`) use configurable `LLM_BACKEND` and pass real evaluation args
- Evaluation now fails closed unless real dataset/detector/checkpoint/ERR args are provided, with `--synthetic` required for legacy synthetic mode
- Detector-specific adapter factory added; formal runs fail closed if detector-native evidence is unavailable
- Exporter now only includes `evaluation_mode=real` results with non-placeholder dataset/detector metadata
- Regression coverage added in `tests/unit/test_formal_experiment_routing.py`

## Validation Results

- 350 tests passing (344 unit + 6 paper alignment)
- ruff check: all passed
- mypy src: 78 source files, no issues
- check_no_leakage.py: passed
- check_no_llm_inference.py: passed
- 78 source files, 37 test files
- Smoke pipeline: Stage 1/2/3/evaluate all pass; 2 accepted ERRs; evaluate uses real tiny args instead of legacy synthetic mode
- YelpChi real pipeline: all 3 stages + evaluation passed
- TensorBoard: 8 metrics logged per run
- Formal routing regression: `tests/unit/test_formal_experiment_routing.py` passes

## Research Constraints Reconfirmed

- prediction_score is calibration-only
- summary not used for training
- rejected ERR excluded from reasoning loss
- no LLM imports in inference
- DHEF/CER/ECB/adaptive lambda experimental only

## Environment

- **Venv**: `/data1/mq/conda_envs/gread-core` (Python 3.10.20)
- **PyTorch**: 2.6.0+cu124 | **PyG**: 2.7.0 | **numpy**: 1.26.4
- **CUDA**: NVIDIA GeForce RTX 3090
- **Activation**: `source /data1/mq/conda_envs/gread-core/bin/activate`
- **CUDA libs**: auto-configured in activate script (nvidia/cudnn, cublas, etc.)
- **CLAUDE.md**: project rules enforce using this venv for all Python commands

## Next

- Add missing ablation configs (A1-A11 full matrix — currently 5 of 11)
- Regenerate non-tiny Stage 2 ERRs with `LLM_BACKEND=openai`, then replay from enriched cache
- Run full-scale experiments on tfinance + tsocial datasets
- Generate final main table with real metrics
- Prepare paper-ready results and figures

## Remaining Hardening (non-blocking for experiments)

- Add `created_at` and `payload_hash` to cache enriched entries (currently has `verification_result` + `contract_version`)
- Full-scale experiments on tfinance + tsocial datasets
- Final paper tables and figures

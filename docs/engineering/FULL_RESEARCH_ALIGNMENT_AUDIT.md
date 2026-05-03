# GReaD-Core Full Repository Audit Report

## 1. Executive Summary

总体状态：**partially aligned / 部分对齐，但存在重大研究对齐缺口**。

当前仓库已经具备较完整的工程外壳：schema、MEP、adapter、trace selection、LLM teacher、deterministic verifier、reasoner、training、inference、evaluation、ablation、reproducibility 文件都存在，`ruff`、unit tests、paper-alignment tests、两个静态 guard 在本次审计中通过。但这不是“可直接做论文评价实验”的状态。关键主路径里仍有多处未被测试覆盖的研究级断裂：Stage 2 CLI 实际没有加载 YAML 风险契约，Stage 3 训练没有使用真实 MEP evidence tokens，inference 对 test-node 子集会崩溃，evaluation 默认用 synthetic/dummy 指标，tri-CEC 与 non-redundancy 不能支撑论文声称的结论。

当前项目成熟度：

| 维度 | 成熟度 | 结论 |
| --- | --- | --- |
| Harness | partial | 基础质量门存在；`mypy src` 失败；CI 也会因 mypy/smoke 失败或 artifact 写入风险不稳定。 |
| Schema / MEP | mostly aligned | `CalibrationChannel.prediction_score` 与 `to_teacher_payload()` 隔离基本正确；但 prompt builder 可接受任意 dict 并泄漏 score。 |
| Verifier | partial | 单元测试中的 YAML verifier 较强；真实 Stage 2 CLI 传入错误 config，导致 contract/label compatibility 主路径失效。 |
| Training | partial | loss mask 函数正确；Stage 3 实际 evidence token 全零且 target slot 用 Python `hash()`，训练信号不可复现且不是 evidence-conditioned。 |
| Inference | misaligned | 无 LLM import；但真实 batch 子集推理会 shape crash，默认 evidence slot 输出不是合法 evidence ID。 |
| Evaluation | misaligned | 检测指标函数存在；evaluate CLI 默认 synthetic/dummy reasoning/CEC，不能支撑 paper evaluation。 |
| Reproducibility | partial | seed/config hash/git commit 有；manifest/checkpoint 缺 split hash、dataset、contract version、ERR cache hash 等关键字段。 |

Top 5 critical issues:

1. **Stage 2 verifier 主路径绕过风险契约**：`src/gread_core/cli/generate_err.py:143-145` 使用 `config.get("verifier", {})` 直接构造 `EvidenceContractVerifier`，没有读取 `configs/contracts/gread_v1.yaml`。探针显示未引用 `detector_signal` 的 `spectral_anomaly` 被接受。
2. **Stage 3 不是有效 evidence-conditioned distillation**：`src/gread_core/training/stage3_train_reasoner.py:201-207` 传入全零 `evidence_token_ids`；`_hash_to_slot()` 使用 Python `hash()`，不同 `PYTHONHASHSEED` 下 target slot 变化。
3. **Inference pipeline 对真实 test-node 子集不可运行**：`src/gread_core/inference/predictor.py:182-201` 使用 detector 返回的全图 embedding 与子集 MEP 拼接；审计探针复现 `Expected size 50 but got size 8`。
4. **Evaluation CLI 当前结果不可信**：`src/gread_core/cli/evaluate.py:329-355` 默认 synthetic evaluation，并为 reasoning metrics 构造固定 dummy evidence/type；`tri_cec` 也使用 dummy MEP。
5. **Adapter 与 evidence vocabulary 不一致且 detector thresholds 硬编码**：adapter `_derive_signal()` 阈值写死，`configs/detectors/*.yaml` 无 evidence thresholds；`degree_level` 产生 `isolated/medium`，但 spec 要 `very_low/low/normal/high/burst/unavailable`。

Top 5 missing features:

1. 真实 Stage 2 contract YAML 加载、schema-only/no-role/no-label ablation gate。
2. 真实 MEP reasoning channel 到 evidence token 的稳定编码与训练/推理共享词表。
3. 真实 inference checkpoint/data path 的 batch-safe 输出测试。
4. 真实 evaluation：reasoning metrics、tri-CEC、non-redundancy 从模型输出和 accepted ERR 读取，而不是 synthetic/dummy。
5. 完整 ablation matrix：当前只有 3 个 config，缺 A1/A2/A3/A6/A7/A9/A10/A11 等关键 ablation。

是否安全进入 evaluate-stage experiments：**否。当前 evaluation results 不可信，不能用于论文表格或 TKDE claim。**

## 2. Command Results

审计命令使用项目 venv：`/data1/mq/conda_envs/gread-core`。审计开始时 `git status --short` 已显示多处既有 dirty changes，包括 `CLAUDE.md`、`pyproject.toml`、`src/gread_core/*`、`configs/experiments/val_real_*.yaml`、`tests/unit/test_real_data_loaders.py` 等；这些不是本次审计产生的，本报告未回滚或修改。

| Command | Status | Notes |
| --- | --- | --- |
| `ruff check .` | pass | 输出 `All checks passed!`。 |
| `mypy src` | fail | 49 errors in 9 files。主要在 `src/gread_core/evaluation/detection.py`、`non_redundancy.py` 的 `np.ndarray` 泛型，`src/gread_core/evidence/generic_signals.py` 的 `torch_scatter` stub 与 Tensor/float assignment，`src/gread_core/data/loaders.py` 的 `scipy.io`/`dgl` stub，`src/gread_core/training/stage2_generate_err.py` 的 `_MEPProxy` 类型，`src/gread_core/cli/evaluate.py:154` 的 `"Tensor" not callable`，TensorBoard `SummaryWriter` typing。 |
| `pytest tests/unit -v` | pass | 323 passed, 2 warnings in 140.96s。warnings：PyG deprecation；`bwgnn_adapter.py:39` variance degrees-of-freedom warning。 |
| `pytest tests/paper_alignment -v` | pass | 6 passed in 1.78s。注意 paper-alignment 覆盖很薄，未覆盖 Stage 2 CLI contract config、real inference、real evaluation。 |
| `python scripts/check_no_leakage.py` | pass | 未发现 exact tokens `prediction_score/fraud_score/base_score/probability_score`；但脚本不检测自然语言 `"prediction score"` 或 `"fraud score"`。 |
| `python scripts/check_no_llm_inference.py` | pass | `src/gread_core/inference` 与 `src/gread_core/models` 无 forbidden imports。 |
| `bash scripts/run_smoke.sh` | not run | 按用户批准的计划不运行：脚本硬编码写入 `artifacts/smoke` 与 `.cache/llm_smoke`，会修改除目标报告外的仓库内容。 |

额外只读探针结果：

| Probe | Result | 结论 |
| --- | --- | --- |
| `EvidenceContractVerifier(config["verifier"])` 验证未引用 required `detector_signal` 的 `spectral_anomaly` | `{'accepted': True, 'reasons': [], 'contract_keys': ['contract_path', 'label_compatibility']}` | Stage 2 CLI 主路径没有加载 YAML contract，contract check 实际失效。 |
| `GReaDInferencePipeline.predict()` 在 tiny graph test nodes 上运行 | `RuntimeError: Sizes of tensors must match... Expected size 50 but got size 8` | 真实子集 inference batch shape 错。 |
| `PromptBuilder().build({... calibration.prediction_score ...})` | `contains_prediction_score=True, contains_0_99=True` | PromptBuilder 自身不强制 teacher payload schema，错误调用可泄漏 score。 |
| `_hash_to_slot()` 两个 `PYTHONHASHSEED` | seed 1: `0 5`；seed 2: `7 2` | evidence target slot mapping 跨进程不稳定。 |

## 3. Spec-to-Code Alignment Matrix

| Spec | Expected | Implemented | Tests | Status | Issues |
| --- | --- | --- | --- | --- | --- |
| `000_constitution` | paper-first、score-blind、offline LLM、deterministic verifier、accepted-only loss、LLM-free inference、tri-CEC/non-redundancy、repro metadata | 大部分文件存在；`configs/default.yaml` 默认禁用实验功能；质量门存在 | unit/paper/guard 存在 | partial | 主路径违反 verifier/evaluation/inference；`mypy` 失败；metadata 不完整。 |
| `001_project_overview_and_claims` | detector-adaptable but not universal；contract-verified score-blind distillation；non-redundancy if proven | README 与 docs 表述基本收缩；adapter 文件覆盖 BWGNN/CARE/PyG/tree | adapter/eval tests 存在 | partial | 当前 evidence adapter 是简化实现；evaluation 不能证明 non-redundancy；不可声称 evaluate results。 |
| `002_score_blind_mep` | `CalibrationChannel.prediction_score`；`ReasoningChannel` 无 score；`to_teacher_payload()` 只输出 reasoning；validators reject forbidden IDs | `src/gread_core/schemas/evidence.py:12-57`、`mep_builder.py:16-86`、`leakage_guard.py:5-21` | `tests/unit/test_mep_score_blind.py`、paper score payload tests、`check_no_leakage.py` | mostly complete | `ReasoningChannel` 不校验 allowed IDs 属于合法 evidence slots；`PromptBuilder.build()` 可接受带 score 的任意 dict。 |
| `003_detector_adapter_protocol` | ABC + BWGNN/CARE/GNN/tree adapters；generic/native/counter evidence；config-driven quantization | `EvidenceAdapter` exists；四类 adapter exist；`quantization.py` 支持 thresholds 参数 | `tests/unit/test_adapter_protocol.py` | partial | detector-native thresholds 写死在 adapter；detector configs 无 thresholds；evidence value vocabulary 与 spec/contract 不一致；GraphSAGE 未实现。 |
| `004_trace_selection` | fixed three buckets + diversity + deterministic seed + prediction_score only selection | `assign_buckets()`、`diversity_sample()`、`TraceSelector` 实现三桶与 diversity | `tests/unit/test_trace_selection.py` | mostly complete | 阈值在函数默认参数中硬编码，未从 config 读取；Stage 2 使用 `_MEPProxy` 导致 mypy fail；no ECB main path。 |
| `005_err_and_llm_teacher` | ERR schema；`training_targets()` excludes summary；prompt uses only `mep.to_teacher_payload()`；cache has prompt/payload hash/model/temp/raw/parsed/verification | `EvidenceRationaleRecord`、`LLMTeacher`、`PromptBuilder`、`PromptCache`、OpenAI/Replay/Stub clients | `test_err_schema.py`、`test_llm_cache.py`、integration replay | partial | cache only stores `prompt_hash` + `response`；no `teacher_payload_hash`、model、temperature、parsed_err、verification_result；PromptBuilder not schema-guarded；OpenAI live allowed only via Stage 2 CLI but not contract-loaded. |
| `006_evidence_contract_verifier` | six hard deterministic checks; YAML contract; required evidence must be cited | Checks exist in `src/gread_core/verification/*`; `contract.py:20-34` requires matching field cited in support | `tests/unit/test_contract_verifier.py` covers important negatives | partial | Unit path OK with YAML, but real CLI passes wrong config. Unknown allowed IDs not directly rejected by schema/availability. `label_noise_mode` config not interpreted beyond forbidden lists. |
| `007_student_reasoner` | EvidenceEncoder; `[z_v; g_v]`; outputs base/final/type/pos/neg; signed masks; rho=0 base | `EvidenceEncoder`、`GReaDReasoner`、heads、residual readout implemented | shape/residual/rho/signed tests | partial | Model architecture exists, but Stage 3 feeds all-zero evidence tokens, so trained model is not actually evidence-conditioned. |
| `008_training_protocol` | Stage1 supervised only；Stage2 LLM only；Stage3 accepted ERR only；main loss formula；summary excluded | Stage1/Stage2/Stage3 modules exist；`ReasoningLoss` masks accepted samples | loss tests and partial integration tests | partial | Stage2 verifier config bypass；Stage3 target hash nondeterministic；Stage3 no real evidence tokens；integration tests not part of required validation and use mock accepted ERRs. |
| `009_inference_protocol` | no LLM import；output node_id/fraud_score/risk/evidence/explanation；deterministic template；batch inference | `PredictionResult` and pipeline exist；guard passes | no-LLM/explanation tests | misaligned | Real subset batch crashes；slot names default to `slot_i` not evidence IDs；risk type sorted order mismatches training index order；no checkpoint-load integration test. |
| `010_evaluation_protocol` | detection metrics, reasoning metrics, tri-CEC on MEP evidence, non-redundancy nested models, JSON outputs | metric modules exist；CLI writes `evaluation_results.json` | unit tests for metrics/CEC/non-red | misaligned | evaluate CLI defaults to synthetic; reasoning metrics dummy; tri-CEC dummy MEP; CEC tokenization can be all zeros and uses Python `hash()`; real evaluation not validated. |
| `011_ablation_matrix` | A1-A11 configs and runner/exporter; paper_warning; split preserved | 3 configs: `ablation_no_reasoner`, `ablation_score_visible`, `ablation_schema_only_verifier`; runner/exporter exist | `test_ablation_configs.py` only checks load/paper_warning/main_gcn_tiny experimental | missing/partial | Most required ablations missing; configs often not wired to behavior (`score_blind: false` has no visible prompt effect); exporter infers dataset/detector incorrectly for real ablation artifacts. |
| `012_reproducibility_and_artifacts` | seed, config hash, split hash, git commit, dataset, contract version, ERR cache hash, checkpoint metadata, table export | `set_seed()` good; registry/checkpoint manager write basic metadata; artifacts exist | metadata tests minimal | partial | Manifest/checkpoint metadata omit split hash, dataset in checkpoint, detector checkpoint path, contract version, ERR cache hash; artifact layout differs from spec. |
| `013_experimental_extensions` | DHEF/CER/ECB/adaptive lambda/prototype/self-consistency/learned verifier disabled by default and under experimental | `configs/default.yaml` disables listed features; `src/gread_core/experimental/__init__.py` only | `test_main_configs_no_experimental` only checks one main config | mostly aligned by absence | Test only checks `main_gcn_tiny.yaml`; no scan of all main/full/val configs; no implementation under experimental, acceptable if deferred. |

## 4. Research Contract Violations

```text
Violation: Stage 2 CLI does not load the Evidence Contract YAML, so accepted ERRs are not contract-verified in the real pipeline.
Severity: critical
File(s): src/gread_core/cli/generate_err.py:143-145, src/gread_core/verification/contract.py:12-15, configs/experiments/smoke_tiny.yaml:60-62
Evidence: CLI passes {"contract_path": "...", "label_compatibility": true}; `check_contract()` expects `config["risk_types"]`. Probe accepted a spectral_anomaly ERR with supporting_evidence=["neighbor_consistency"] and no required detector_signal citation.
Why it violates the research design: Accepted ERR requires risk-evidence contract consistency and label compatibility. In real Stage 2, contract rules are effectively absent.
Recommended fix: Load `contract_path` YAML in Stage 2 CLI/training factory; merge explicit ablation overrides after loading; fail closed if contract missing.
Suggested test to add: CLI-level or `generate_errs()` integration test where `detector_signal=high_frequency_response_high` but ERR omits `detector_signal`; assert rejected when using a normal experiment config.
```

```text
Violation: PromptBuilder can leak `prediction_score` if called with a non-teacher payload.
Severity: high
File(s): src/gread_core/llm/prompt_builder.py:24-31
Evidence: Probe `PromptBuilder().build({"calibration":{"prediction_score":0.99}, ...})` produced prompt containing `prediction_score` and `0.99`.
Why it violates the research design: The only allowed LLM input should be `MinimalEvidencePackage.to_teacher_payload()`. A public prompt builder accepting arbitrary dict is a bypass.
Recommended fix: Change `PromptBuilder.build()` to accept `MinimalEvidencePackage` or validate exact top-level keys `{node_id, detector_name, reasoning}` and reject calibration/score tokens recursively.
Suggested test to add: `test_prompt_builder_rejects_calibration_or_prediction_score_payload`.
```

```text
Violation: Stage 3 training does not consume actual MEP reasoning evidence tokens.
Severity: critical
File(s): src/gread_core/training/stage3_train_reasoner.py:201-207
Evidence: The forward call passes `torch.zeros(num_samples, num_evidence_slots, ...)` for every sample.
Why it violates the research design: The student is supposed to learn from `[z_v; g_v]` where `g_v=phi(E_v)`. With all-zero evidence tokens, evidence-conditioned residual reasoning is not trained on MEP evidence.
Recommended fix: Persist each accepted ERR with its MEP reasoning payload or stable evidence token IDs from Stage 2; build Stage 3 batches from those tokens.
Suggested test to add: Stage 3 integration test with two accepted ERRs having different MEP reasoning values; assert non-identical `evidence_token_ids` enter the reasoner.
```

```text
Violation: Evidence target slot mapping is nondeterministic across processes.
Severity: high
File(s): src/gread_core/training/stage3_train_reasoner.py:109-123, src/gread_core/evaluation/cec.py:109-114
Evidence: `_hash_to_slot("detector_signal",16)` returned `0` under `PYTHONHASHSEED=1` and `7` under `PYTHONHASHSEED=2`.
Why it violates the research design: Reproducible evidence masks require stable evidence slot mapping. Python `hash()` is salted per process by default.
Recommended fix: Replace with a fixed ordered evidence vocabulary or SHA256-derived deterministic mapping stored in config/metadata.
Suggested test to add: Spawn subprocesses with two `PYTHONHASHSEED` values and assert evidence ID to slot mapping identical.
```

```text
Violation: LLM cache does not satisfy required reproducibility schema.
Severity: high
File(s): src/gread_core/llm/cache.py:20-66, src/gread_core/llm/teacher.py:90-97
Evidence: `PromptCache.put()` writes only {"prompt_hash": key, "response": response}.
Why it violates the research design: Spec requires prompt hash, teacher payload hash, model name, temperature, raw response, parsed ERR, verification result, created_at, prompt template version, contract version.
Recommended fix: Introduce structured cache records and have `LLMTeacher` persist parsed/verified results, not raw response only.
Suggested test to add: Cache JSONL schema test asserting required keys and replay without client call.
```

```text
Violation: Inference pipeline is not batch-safe for target node subsets.
Severity: critical
File(s): src/gread_core/inference/predictor.py:182-201, src/gread_core/detectors/pyg_gnn.py:68-93
Evidence: Detector returns logits for mask-selected nodes but embeddings for all nodes. Probe with 50-node tiny graph and 8 test nodes failed with tensor size mismatch.
Why it violates the research design: Inference must work LLM-free and output per-node fraud reasoning. Current CLI chooses test nodes, so real inference can crash.
Recommended fix: Strip masks and compute all-node logits/embeddings, then index both by `node_ids`; or pass a graph view with exactly target nodes and consistent embeddings/logits.
Suggested test to add: End-to-end `GReaDInferencePipeline.predict(data, test_node_ids)` asserting all required fields and `len(outputs)==len(node_ids)`.
```

```text
Violation: Inference evidence output is not contract evidence IDs by default.
Severity: high
File(s): src/gread_core/inference/predictor.py:90-94, 224-241
Evidence: Default `slot_names` are `slot_0...slot_K`; supporting/counter evidence are mask positions, not `degree_level`, `detector_signal`, etc.
Why it violates the research design: Inference must output supporting_evidence/counter_evidence grounded in evidence IDs. `slot_i` is not valid contract evidence.
Recommended fix: Define a canonical evidence slot vocabulary in config/schema and use it consistently in encoder, training targets, inference masks, evaluation.
Suggested test to add: Inference output evidence IDs are subset of canonical `EVIDENCE_SLOTS` and never include `prediction_score`.
```

```text
Violation: Risk type index mapping differs between training and inference.
Severity: high
File(s): src/gread_core/training/stage3_train_reasoner.py:26-34, src/gread_core/inference/predictor.py:31-32
Evidence: Training uses insertion mapping where `spectral_anomaly -> 2`; inference uses `sorted(RISK_TYPES)`, changing indices.
Why it violates the research design: Predicted `risk_type` can be decoded to the wrong label at inference/evaluation.
Recommended fix: Export one canonical ordered risk taxonomy list and import it everywhere.
Suggested test to add: For each risk type string, training index -> inference label round-trip must match.
```

```text
Violation: Evaluation CLI uses synthetic/dummy reasoning and CEC by default.
Severity: critical
File(s): src/gread_core/cli/evaluate.py:329-355, 399-431
Evidence: Without dataset/detector/detector-checkpoint args, evaluation uses `_generate_synthetic_data`. Reasoning metrics use fixed `e0/e1/e2` and fixed `camouflage_neighbor`; tri-CEC uses `dummy_meps`.
Why it violates the research design: Evaluation must measure detection/reasoning/CEC/non-redundancy on model outputs and ERR references. Current results are not evidence of GReaD-Core alignment.
Recommended fix: Make synthetic mode explicit and excluded from paper tables; require real checkpoint/data/ERR artifacts for paper evaluation; compute reasoning metrics from predictions and accepted ERR references.
Suggested test to add: CLI evaluation smoke test that fails if paper mode tries to use synthetic or dummy references.
```

```text
Violation: Tri-CEC implementation can be disconnected from actual MEP evidence encoding.
Severity: high
File(s): src/gread_core/evaluation/cec.py:89-118, src/gread_core/cli/evaluate.py:406-430
Evidence: CLI passes `slot_to_id = {"slot_i": i}`; `_mep_to_evidence_token_ids()` checks those keys against reasoning dict keys, producing all-zero IDs. The smoke artifact tri-CEC is all 0.0.
Why it violates the research design: CEC weakening must operate on MEP reasoning channel and test model response. All-zero tokenization makes weakening invisible.
Recommended fix: Use canonical evidence field/value vocabulary and pass real MEPs from evaluation samples.
Suggested test to add: CEC test where weakening `detector_signal` changes at least one token ID and leaves `prediction_score` unchanged.
```

```text
Violation: Detector adapters use hard-coded detector-native thresholds.
Severity: medium
File(s): src/gread_core/adapters/bwgnn_adapter.py:42-50, caregnn_adapter.py:41-49, pyg_gnn_adapter.py:53-61, tree_adapter.py:34-40; configs/detectors/*.yaml
Evidence: Thresholds like `ratio >= 0.6`, `cos_dist >= 0.4`, `variance >= 0.05` are embedded in code; detector configs do not define evidence thresholds.
Why it violates the research design: Spec requires continuous signals quantized through config-driven thresholds.
Recommended fix: Add detector evidence threshold sections to configs and pass them into adapters.
Suggested test to add: Adapter output changes when threshold config changes; no adapter threshold constants except defaults in config loader.
```

```text
Violation: Required ablation matrix is mostly absent or not wired.
Severity: high
File(s): configs/experiments/ablation_*.yaml, src/gread_core/evaluation/ablation.py, scripts/run_ablations.sh
Evidence: Only `ablation_no_reasoner`, `ablation_score_visible`, and `ablation_schema_only_verifier` exist. `score_blind: false` is not consumed by prompt/MEP code.
Why it violates the research design: Paper requires A1-A11 ablations to isolate score-blindness, verifier checks, signed evidence, diversity, adapters, etc.
Recommended fix: Add missing configs and implement config gates for each ablation.
Suggested test to add: Parametrized test asserting every required ablation exists, has `paper_warning`, preserves split config, and toggles a real behavior.
```

```text
Violation: Reproducibility metadata omits required fields.
Severity: medium
File(s): src/gread_core/experiment/registry.py:92-104, src/gread_core/training/checkpointing.py:69-77, artifacts/smoke/stage*/epoch_0005/metadata.json
Evidence: Manifests/checkpoints include experiment_id/git/config_hash/seed/stage/timestamp, but not split_hash, detector checkpoint path, contract version, ERR cache hash, dataset in checkpoint metadata.
Why it violates the research design: Experiments cannot be traced to split/cache/contract versions required by spec 012.
Recommended fix: Extend registry/checkpoint metadata and write split/ERR cache hashes during each stage.
Suggested test to add: Checkpoint and manifest schema tests against spec 012 required fields.
```

```text
Violation: Real dataset loader silently falls back to synthetic data.
Severity: high
File(s): src/gread_core/data/loaders.py:169-212
Evidence: `except (FileNotFoundError, ImportError, Exception)` logs warning and returns synthetic fallback for real dataset names.
Why it violates the research design: Evaluation can appear to run on YelpChi/Amazon/T-Finance/T-Social while actually using synthetic data, invalidating paper results.
Recommended fix: Default fail closed for real dataset configs; allow synthetic fallback only with explicit config flag.
Suggested test to add: Missing real dataset path raises unless `allow_synthetic_fallback: true`.
```

## 5. Completed Components

Harness:

- **partial**. `pyproject.toml` has ruff/mypy/pytest config; `.github/workflows/ci.yml`, `paper_alignment.yml`, `smoke.yml` exist; scripts include `check_no_leakage.py`, `check_no_llm_inference.py`, `run_smoke.sh`.
- Tests: `ruff` pass, unit/paper pass, guards pass.
- Gap: `mypy src` fails; smoke not safe under read-only constraint; CI likely unstable because `mypy src` strict fails.

Schemas:

- **mostly complete**. `CalibrationChannel`, `ReasoningChannel`, `MinimalEvidencePackage`, `EvidenceRationaleRecord` implemented in `src/gread_core/schemas/evidence.py` and `err.py`.
- Tests: `tests/unit/test_mep_score_blind.py`, `tests/unit/test_err_schema.py`.
- Gap: evidence slot vocabulary not validated; risk taxonomy order duplicated/inconsistent.

MEP:

- **mostly complete**. `build_mep()` puts `prediction_score` only in `CalibrationChannel`; `to_teacher_payload()` excludes calibration.
- Tests: score-blind MEP and prompt payload tests pass.
- Gap: prompt builder bypass can leak score; evidence values are not constrained to contract vocabulary.

Adapter:

- **partial**. ABC and BWGNN/CARE/PyG/tree adapters exist; generic signals computed.
- Tests: adapter protocol tests pass.
- Gap: hard-coded thresholds; simplified detector-native signals; no GraphSAGE; config-driven quantization not wired.

Trace selection:

- **mostly complete**. Three buckets, deterministic seed, evidence diversity sampling, budget fractions exist.
- Tests: `tests/unit/test_trace_selection.py`.
- Gap: thresholds hard-coded; no metadata-rich selection logs beyond bucket/diversity arrays.

LLM teacher:

- **partial**. LLM code isolated under `src/gread_core/llm`; OpenAI/Replay/Stub clients; cache/replay tests.
- Gap: cache schema incomplete; malformed JSON parse retry limited to fence stripping; Stage 2 CLI verifier config bypass.

Verifier:

- **partial**. Rule-based deterministic checks exist; YAML contract tests pass; no LLM/learned verifier in main verifier.
- Gap: real Stage 2 path not YAML-driven; unknown evidence IDs can be allowed in MEP allowed lists; label-noise mode not fully expressed.

Reasoner:

- **partial**. Evidence encoder, signed heads, residual readout, rho ablation implemented.
- Tests: shape, residual, rho, no-LLM pass.
- Gap: training path does not feed real evidence tokens.

Loss/training:

- **partial**. `ReasoningLoss` implements masked accepted loss correctly; Stage 1/2/3 modules exist.
- Tests: loss masking tests pass.
- Gap: Stage 2 verifier wrong config; Stage 3 evidence encoding/targets broken for research use.

Inference:

- **misaligned**. No LLM imports and deterministic explanation template exist.
- Tests: no-LLM and template tests pass.
- Gap: real sub-batch inference crashes; evidence/risk decoding invalid.

Evaluation:

- **misaligned**. Metric helper functions exist.
- Tests: detection/reasoning/non-redundancy/CEC helper tests pass.
- Gap: CLI evaluation uses synthetic/dummy references; paper metrics not meaningful.

Ablations:

- **partial/missing**. Three config files and runner/exporter exist.
- Gap: required ablation matrix incomplete and many toggles not wired.

Reproducibility:

- **partial**. `set_seed()`, config hash, git commit, basic manifests/checkpoints exist.
- Gap: missing split hash, dataset split metadata, contract version, ERR cache hash, detector checkpoint path.

## 6. Missing or Incomplete Components

P0 = blocks research correctness:

1. Fix Stage 2 contract YAML loading and fail-closed verifier construction.
2. Implement stable canonical evidence vocabulary shared by MEP, verifier, training targets, model, inference, CEC.
3. Replace Stage 3 all-zero evidence tokens with persisted MEP reasoning tokens.
4. Fix inference target-node indexing and risk/evidence decoding.
5. Replace evaluation synthetic/dummy paper path with real artifact-based evaluation.

P1 = blocks experiments:

1. Make `mypy src` pass under configured strict mode.
2. Make smoke runnable without overwriting existing artifacts or document isolated output option.
3. Remove silent synthetic fallback for real datasets or require explicit fallback flag.
4. Write complete manifest/checkpoint metadata.
5. Implement cache schema with payload/model/verification hashes.

P2 = weakens paper claim:

1. Add missing ablations A1-A11.
2. Make adapter thresholds config-driven and detector-specific.
3. Add GraphSAGE or narrow claims to implemented PyG GCN/GAT.
4. Add stronger negative tests for unknown allowed evidence IDs and prompt builder misuse.
5. Add real tri-CEC/non-redundancy regression tests.

P3 = nice-to-have:

1. Improve explanation template quality after correctness is fixed.
2. Add richer selection metadata and artifact logging.
3. Refactor duplicated detector construction code in CLIs.

## 7. Vulnerability and Leakage Analysis

prediction_score leakage risk:

- Low in `MinimalEvidencePackage.to_teacher_payload()` and `LLMTeacher.generate_err()` normal path: `src/gread_core/schemas/evidence.py:53-58`, `src/gread_core/llm/teacher.py:56-57`.
- High through `PromptBuilder.build()` misuse: it accepts arbitrary dict and can render calibration/prediction_score.
- `configs/experiments/ablation_score_visible.yaml` declares score-visible ablation, but code does not appear to implement a controlled score-visible path; current ablation likely not meaningful.

LLM-in-inference risk:

- Static import risk is low: `check_no_llm_inference.py` passed; model/inference packages do not import `gread_core.llm`, `openai`, `requests`, `httpx`.
- Functional inference risk is high for a different reason: current pipeline can crash before producing outputs.

summary-as-label risk:

- Low in schema/loss: `EvidenceRationaleRecord.training_targets()` excludes summary; `ReasoningLoss.forward()` has no summary parameter; Stage 3 uses `err["risk_type"]`, support/counter evidence only.
- Continue to guard because Stage 2 cache currently stores raw responses without structured field separation.

rejected-ERR loss leakage risk:

- Low inside `ReasoningLoss`: accepted mask works and unit tests pass.
- Medium in pipeline semantics: Stage 2 only saves accepted ERRs with full ERR content, and rejected records lack rejection reasons; if verifier config is wrong, bad ERRs become accepted and enter loss.

weak verifier bypass risk:

- Critical in real Stage 2 CLI due contract YAML not loaded.
- Medium in schema/availability: MEP allowed lists can include unknown evidence IDs; availability treats allowed unknown IDs as available, so rejection may depend accidentally on contract failure.

CEC disconnected rationale risk:

- Critical. Stage 3 trains with all-zero evidence tokens; CEC tokenization uses `slot_i` mapping and Python hash, so weakening may not affect model input. Smoke artifact reports `score_cec=type_cec=evidence_cec=0.0`.

experimental feature accidentally enabled risk:

- Main/default configs inspected show experimental flags disabled. Risk is low for current code.
- Test coverage weak: `test_main_configs_no_experimental()` only checks `main_gcn_tiny.yaml`, not all main/full/val configs.

Security and robustness:

- `scripts/run_validation.sh` and `scripts/run_full_experiments.sh` contain `rm -rf "$OUTPUT_DIR"`; acceptable only with explicit run intent, not safe for audit.
- `src/gread_core/data/loaders.py` has hard-coded absolute data root `/data1/mq/.../PriorF-GNN/datasets`.
- Broad exception swallowing/fallback in data loaders and exporter can hide missing real data/config.
- No obvious secrets found in scanned configs/src; untracked files such as `=4.10.0` and tmux logs existed before this audit and should be reviewed separately.

## 8. Evaluation-Stage Readiness

1. Is the project truly ready for evaluation? **No.**
2. Are metrics implemented correctly? **Detection helper functions mostly yes; evaluation CLI usage no.**
3. Are tri-CEC and non-redundancy implemented? **Helpers exist, but paper-use path is not correctly connected to real evidence/model outputs.**
4. Are ablations implemented? **No; only 3 partial configs exist and key toggles are not wired.**
5. Are artifacts reproducible? **Partially; metadata is insufficient and some mappings use nondeterministic `hash()`.**
6. Are current evaluation results trustworthy? **No. Synthetic/dummy evaluation and silent dataset fallback make current artifacts unsuitable for paper claims.**

Evaluation readiness verdict: **Not ready; major alignment issues**.

## 9. Recommended Next Tasks

```text
Task: Fix Stage 2 verifier contract loading.
Priority: P0
Files to inspect/change: src/gread_core/cli/generate_err.py, src/gread_core/training/stage2_generate_err.py, src/gread_core/verification/verifier.py, configs/contracts/gread_v1.yaml
Why: Current accepted ERRs are not guaranteed contract-consistent in the real CLI path.
Acceptance criteria: Normal config loads YAML risk_types; missing contract fails closed; schema-only/no-label ablations intentionally override checks.
Tests to add/run: CLI-level verifier negative test; pytest tests/unit/test_contract_verifier.py; pytest tests/paper_alignment.
```

```text
Task: Create canonical evidence/risk vocabulary and stable tokenization.
Priority: P0
Files to inspect/change: src/gread_core/schemas/risk_taxonomy.py, src/gread_core/models/evidence_encoder.py, src/gread_core/training/stage3_train_reasoner.py, src/gread_core/inference/predictor.py, src/gread_core/evaluation/cec.py
Why: Training, inference, and CEC currently use inconsistent or nondeterministic mappings.
Acceptance criteria: Same evidence ID maps to same slot across processes; risk type index order is identical in training/inference/evaluation.
Tests to add/run: subprocess hash-seed test; risk type round-trip test; inference evidence ID subset test.
```

```text
Task: Persist and consume real MEP reasoning tokens in Stage 3.
Priority: P0
Files to inspect/change: src/gread_core/training/stage2_generate_err.py, src/gread_core/training/stage3_train_reasoner.py, src/gread_core/llm/cache.py
Why: Current Stage 3 uses all-zero evidence tokens, so the reasoner is not evidence-conditioned.
Acceptance criteria: accepted_errs include MEP reasoning payload or evidence_token_ids; Stage 3 batch has nonzero distinct tokens for distinct evidence.
Tests to add/run: Stage 2/3 tiny integration test verifying actual token diversity; loss masking tests.
```

```text
Task: Fix LLM-free inference data flow.
Priority: P0
Files to inspect/change: src/gread_core/inference/predictor.py, src/gread_core/cli/infer.py, detectors forward_with_embedding contract
Why: Current pipeline crashes for test-node subsets and outputs invalid slot_i evidence.
Acceptance criteria: Batch inference works for arbitrary node_ids; outputs required fields; evidence IDs are canonical; no LLM imports.
Tests to add/run: new inference integration test on tiny graph test_mask; scripts/check_no_llm_inference.py.
```

```text
Task: Replace evaluation CLI dummy path with real artifact-based evaluation.
Priority: P0
Files to inspect/change: src/gread_core/cli/evaluate.py, src/gread_core/evaluation/reasoning.py, src/gread_core/evaluation/cec.py, src/gread_core/evaluation/non_redundancy.py
Why: Current evaluation artifacts cannot support TKDE claims.
Acceptance criteria: Paper evaluation requires dataset/detector/checkpoints/ERR references; synthetic mode is explicit and labeled non-paper; tri-CEC uses real MEPs.
Tests to add/run: evaluation smoke with real tiny artifacts in temp dir; CEC token-change test; non-redundancy output schema test.
```

```text
Task: Complete ablation matrix.
Priority: P1
Files to inspect/change: configs/experiments/ablation_*.yaml, src/gread_core/evaluation/ablation.py, scripts/run_ablations.sh, scripts/export_results.py
Why: Required design-choice isolation is missing.
Acceptance criteria: A1-A11 configs exist, have paper_warning, preserve split, and trigger real behavior changes.
Tests to add/run: parametrized ablation config test covering all required ablations.
```

```text
Task: Make reproducibility metadata spec-complete.
Priority: P1
Files to inspect/change: src/gread_core/experiment/registry.py, src/gread_core/training/checkpointing.py, src/gread_core/data/splits.py, src/gread_core/llm/cache.py
Why: Current artifacts cannot fully reproduce split/cache/contract state.
Acceptance criteria: manifest/checkpoint include dataset, split_hash, detector checkpoint, contract_version, ERR cache hash, software versions.
Tests to add/run: schema tests for manifest and checkpoint metadata.
```

```text
Task: Fix strict typing quality gate.
Priority: P1
Files to inspect/change: pyproject.toml, src/gread_core/evaluation/detection.py, non_redundancy.py, data/loaders.py, evidence/generic_signals.py, training/stage2_generate_err.py, cli/evaluate.py
Why: `mypy src` is required by AGENTS.md and CI.
Acceptance criteria: `mypy src` exits 0 without disabling major error categories.
Tests to add/run: mypy src; ruff check .
```

## 10. Minimal Fix Plan

Day 1 fixes:

- Load verifier YAML contract in Stage 2 and add fail-closed config handling.
- Add canonical ordered `RISK_TYPES_ORDERED` and `EVIDENCE_SLOTS_ORDERED`; replace sorted risk type decode and Python `hash()`.
- Persist MEP reasoning payload/token IDs with accepted ERRs and feed them into Stage 3 instead of zeros.
- Add tests for Stage 2 contract rejection, risk type round-trip, stable evidence slot mapping, and Stage 3 nonzero evidence tokens.

Day 2 fixes:

- Fix inference indexing for node subsets and output canonical evidence IDs.
- Make evaluate CLI require real inputs for paper mode; label synthetic mode as non-paper and exclude from exporter by default.
- Replace dummy tri-CEC MEPs with real MEPs; ensure weakening changes token IDs but not `prediction_score`.
- Extend metadata with split hash, contract version, ERR cache hash, dataset/checkpoint paths.
- Add missing high-priority ablation configs or clearly mark them unavailable.

Validation commands:

```bash
/data1/mq/conda_envs/gread-core/bin/ruff check .
/data1/mq/conda_envs/gread-core/bin/mypy src
/data1/mq/conda_envs/gread-core/bin/pytest tests/unit -v
/data1/mq/conda_envs/gread-core/bin/pytest tests/paper_alignment -v
/data1/mq/conda_envs/gread-core/bin/python scripts/check_no_leakage.py
/data1/mq/conda_envs/gread-core/bin/python scripts/check_no_llm_inference.py
```

Training/model changes后，再运行 smoke，但建议先把脚本改为可指定临时 output/cache，避免覆盖现有 artifacts：

```bash
bash scripts/run_smoke.sh
```

## 11. Do-Not-Fix-Yet List

以下内容应继续保持 experimental 或 deferred，不应混入主方法：

- DHEF
- CER as training regularizer
- evidence-conflict bucket
- adaptive lambda
- prototype prompting
- multi-sample LLM self-consistency
- learned verifier

当前更应先修主方法闭环：score-blind MEP -> offline ERR -> YAML contract verifier -> accepted-only stable evidence distillation -> LLM-free inference -> real evaluation。

## 12. Final Verdict

```text
Final verdict:
- Research alignment: partially aligned, with critical main-path violations in Stage 2 verifier, Stage 3 evidence conditioning, inference, and evaluation.
- Engineering quality: mixed; ruff/unit/paper guards pass, but mypy fails and several tests only cover happy paths or helper functions.
- Evaluation readiness: Not ready; major alignment issues.
- Most dangerous current flaw: Stage 2 CLI does not load the YAML Evidence Contract Verifier rules, so rejected-by-paper ERRs can become accepted training targets.
- First fix to make: Fix contract YAML loading/fail-closed verifier construction, then add a CLI-level negative test proving required evidence must be cited before ERR enters Stage 3.
```

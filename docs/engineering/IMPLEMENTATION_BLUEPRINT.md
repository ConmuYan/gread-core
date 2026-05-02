下面这份可以直接作为 **GReaD-Core 的 Agentic Engineering / Harness Engineering 实现蓝图**，用于 Claude Code、Codex、Cursor、Windsurf、Gemini CLI 等主流 AI coding agent。核心思想是：**不要让 AI Agent “自由发挥写科研代码”，而是用 spec、接口、测试、CI、验收准则、禁止项和任务切片把它锁进研究方案的轨道里。**

这套方案严格对齐上一版最终研究方案：**Score-blind MEP → Detector-Evidence Adapter → Evidence Contract Verifier → Offline LLM ERR → Evidence-conditioned residual reasoner → LLM-free inference → tri-CEC / non-redundancy evaluation**。你上传的多 AI 评审意见中也集中攻击了 verifier 不够硬、score leakage、CEC 断裂、训练流程与跨 detector 可比性，这些都会被下面的代码 harness 明确约束住。

---

# 1. 总体开发范式：Harness-guided Vibe Coding

这里的 “Vibe Coding” 不能理解为“自然语言随便让 AI 写代码”。面向 TKDE 研究代码，正确范式应该是：

> **人类定义研究契约、软件契约和验证闭环；AI Agent 在契约内高速实现。**

这正是 OpenAI 所说的 harness engineering：工程师的工作从亲手写每一行代码，转为设计环境、指定意图、构建反馈回路，让 agent 可靠执行；OpenAI 的实践也强调 “Humans steer. Agents execute.”，并把 repository knowledge 作为系统事实来源。([OpenAI][1])

对于 GReaD-Core，harness 由四类东西组成：

1. **Feedforward guides**：`AGENTS.md`、`CLAUDE.md`、Spec Kit specs、architecture docs、YAML config、Pydantic schemas。
2. **Feedback sensors**：pytest、ruff、mypy、smoke training、no-leakage tests、no-LLM-inference tests、contract verifier tests。
3. **Execution rails**：issue 切片、branch strategy、definition of done、PR checklist、experiment registry。
4. **Research guards**：任何代码改动都必须能映射回论文模块；不能引入与论文不一致的“额外聪明模块”。

GitHub Spec Kit 很适合这个项目，因为它把开发拆成 specify、plan、tasks、implement 四个阶段，并让 spec 成为 AI coding agent 的 source of truth；GitHub 官方说明中也强调，spec-driven development 的目标是减少“看起来能跑但不符合意图”的 vibe-coding 失败模式。([The GitHub Blog][2])

---

# 2. 推荐工具链与参考仓库

## 2.1 Agent / Harness 工具

| 类型                   | 推荐                                       | 用途                                                                                                                                                                                              |
| -------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spec-driven workflow | GitHub Spec Kit                          | 生成 constitution、spec、plan、task list，适配多个 coding agent。Spec Kit 官方仓库说明它支持 30+ AI coding agents，并提供 `/speckit.constitution`、`/speckit.specify`、`/speckit.plan`、`/speckit.tasks` 等命令。([GitHub][3]) |
| Codex 执行             | OpenAI Codex CLI                         | 终端本地 coding agent。OpenAI Codex 仓库说明 Codex CLI 是在本地电脑运行的轻量 coding agent。([GitHub][4])                                                                                                            |
| Codex 指令文件           | `AGENTS.md`                              | OpenAI 官方文档说明 Codex 会按全局、项目和子目录层级读取 `AGENTS.md` / `AGENTS.override.md`，并从 root 到当前目录合并指令。([OpenAI 开发者][5])                                                                                      |
| Claude 执行            | Claude Code + Skills                     | Claude 官方文档说明 Agent Skills 是以 `.claude/skills/<skill>/SKILL.md` 形式存在的文件系统 artifact，并可被自动发现、按上下文触发。([Claude][6])                                                                                 |
| Skills 社区库           | `alirezarezvani/claude-skills`           | 跨 Claude Code、OpenAI Codex、Gemini CLI、Cursor 等多种 agent 的 skill/plugin 集合，可借鉴其 skill 组织方式。([GitHub][7])                                                                                          |
| Claude Code 最佳实践     | `shanraisshan/claude-code-best-practice` | 参考 agent、commands、skills、workflows、hooks、CLAUDE.md 等组织模式。([GitHub][8])                                                                                                                          |
| Harness 参考           | `ai-boost/awesome-harness-engineering`   | 汇总 harness engineering 的工具、模式、eval、memory、MCP、permissions、observability 和 orchestration 资源。([GitHub][9])                                                                                        |

## 2.2 图欺诈检测与科研代码参考

| 类型        | 推荐                                         | 用途                                                                                                                                                                     |
| --------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 图异常检测库    | PyGOD                                      | PyGOD 基于 PyTorch Geometric 和 PyTorch，提供 10+ 图异常检测算法、统一 API、mini-batch / sampling 支持，适合作为工程风格参考。([GitHub][10])                                                          |
| Benchmark | GADBench                                   | GADBench 提供 Reddit、Weibo、Amazon、YelpChi 等监督图异常检测数据集信息与基准设置，适合作为评估与数据接口参考。([GitHub][11])                                                                                |
| BWGNN     | `squareRoot3/Rethinking-Anomaly-Detection` | BWGNN 官方实现，README 说明其已被集成到 GADBench，并给出 Yelp、Amazon、T-Finance、T-Social 等运行入口。([GitHub][12])                                                                            |
| CARE-GNN  | `YingtongDou/CARE-GNN`                     | CARE-GNN 官方实现，包含 label-aware similarity、similarity-aware neighbor selector、relation-aware neighbor aggregator 三个核心模块，可用于 camouflage detector adapter 参考。([GitHub][13]) |
| 论文资源      | `safe-graph/graph-fraud-detection-papers`  | 图/Transformer 欺诈、异常、离群检测论文与工具箱列表，包含 LLM、GNN、toolbox、dataset、survey 等分区。([GitHub][14])                                                                                  |

---

# 3. 代码仓库目标形态

推荐仓库名：

```text
gread-core
```

推荐定位：

```text
A contract-verified, score-blind evidence distillation framework for LLM-free graph fraud reasoning.
```

核心开发原则：

```text
1. Paper-first: every module must map to a paper component.
2. Config-first: no hidden constants in code.
3. Test-first for contracts: verifier, leakage, inference purity must be unit-tested before model training.
4. LLM offline only: no LLM package import in inference path.
5. Score-blind by construction: prediction_score cannot enter teacher prompts or evidence labels.
6. Detector-adaptable, not detector-universal.
7. Reproducible experiments: seed, config, commit hash, dataset split, teacher cache hash all logged.
```

---

# 4. 推荐目录结构

```text
gread-core/
├── AGENTS.md
├── CLAUDE.md
├── pyproject.toml
├── README.md
├── LICENSE
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── smoke.yml
│   │   └── paper_alignment.yml
│   └── ISSUE_TEMPLATE/
│       ├── feature_task.md
│       ├── experiment_task.md
│       └── bug_report.md
├── .claude/
│   ├── skills/
│   │   └── gread-core-implementer/
│   │       └── SKILL.md
│   └── commands/
│       ├── implement-gread-module.md
│       ├── verify-research-alignment.md
│       └── run-paper-smoke.md
├── specs/
│   ├── constitution.md
│   ├── 000_overview.md
│   ├── 001_score_blind_mep.md
│   ├── 002_detector_adapter_protocol.md
│   ├── 003_evidence_contract_verifier.md
│   ├── 004_llm_teacher_and_err.md
│   ├── 005_student_reasoner.md
│   ├── 006_training_protocol.md
│   ├── 007_evaluation_protocol.md
│   └── 008_ablation_matrix.md
├── configs/
│   ├── default.yaml
│   ├── datasets/
│   │   ├── yelpchi.yaml
│   │   ├── amazon.yaml
│   │   ├── tfinance.yaml
│   │   └── tsocial.yaml
│   ├── detectors/
│   │   ├── bwgnn.yaml
│   │   ├── caregnn.yaml
│   │   ├── gcn.yaml
│   │   ├── gat.yaml
│   │   └── tree_neighbor.yaml
│   ├── contracts/
│   │   └── gread_v1.yaml
│   ├── prompts/
│   │   └── err_generation.yaml
│   └── experiments/
│       ├── main_bwgnn_yelp.yaml
│       ├── main_bwgnn_amazon.yaml
│       ├── ablation_score_visible.yaml
│       ├── ablation_schema_only_verifier.yaml
│       └── ablation_parallel_heads.yaml
├── src/
│   └── gread_core/
│       ├── __init__.py
│       ├── schemas/
│       │   ├── evidence.py
│       │   ├── err.py
│       │   ├── risk_taxonomy.py
│       │   └── experiment.py
│       ├── data/
│       │   ├── loaders.py
│       │   ├── splits.py
│       │   ├── transforms.py
│       │   └── pyg_dgl_bridge.py
│       ├── detectors/
│       │   ├── base.py
│       │   ├── pyg_gnn.py
│       │   ├── bwgnn.py
│       │   ├── caregnn.py
│       │   └── tree_neighbor.py
│       ├── evidence/
│       │   ├── mep.py
│       │   ├── encoder.py
│       │   ├── quantization.py
│       │   ├── generic_signals.py
│       │   └── leakage_guard.py
│       ├── adapters/
│       │   ├── base.py
│       │   ├── bwgnn_adapter.py
│       │   ├── caregnn_adapter.py
│       │   ├── pyg_gnn_adapter.py
│       │   └── tree_adapter.py
│       ├── tracing/
│       │   ├── selector.py
│       │   ├── diversity.py
│       │   └── buckets.py
│       ├── llm/
│       │   ├── teacher.py
│       │   ├── clients.py
│       │   ├── prompt_builder.py
│       │   ├── cache.py
│       │   └── templates/
│       │       └── err_generation.j2
│       ├── verification/
│       │   ├── schema.py
│       │   ├── contract.py
│       │   ├── label_compatibility.py
│       │   ├── role_consistency.py
│       │   └── verifier.py
│       ├── models/
│       │   ├── evidence_encoder.py
│       │   ├── reasoner.py
│       │   ├── heads.py
│       │   └── residual_readout.py
│       ├── losses/
│       │   ├── supervised.py
│       │   └── reasoning.py
│       ├── training/
│       │   ├── stage1_train_detector.py
│       │   ├── stage2_generate_err.py
│       │   ├── stage3_train_reasoner.py
│       │   ├── checkpointing.py
│       │   └── trainer.py
│       ├── evaluation/
│       │   ├── detection.py
│       │   ├── reasoning.py
│       │   ├── cec.py
│       │   ├── non_redundancy.py
│       │   └── ablations.py
│       ├── inference/
│       │   ├── predictor.py
│       │   ├── explanation_template.py
│       │   └── no_llm_guard.py
│       ├── experiment/
│       │   ├── logger.py
│       │   ├── registry.py
│       │   └── seed.py
│       └── cli/
│           ├── train_detector.py
│           ├── generate_err.py
│           ├── train_reasoner.py
│           ├── evaluate.py
│           └── infer.py
├── tests/
│   ├── unit/
│   │   ├── test_mep_score_blind.py
│   │   ├── test_err_schema.py
│   │   ├── test_contract_verifier.py
│   │   ├── test_label_compatibility.py
│   │   ├── test_no_llm_inference.py
│   │   ├── test_loss_masking.py
│   │   └── test_explanation_template.py
│   ├── integration/
│   │   ├── test_stage1_stage2_stage3_tiny.py
│   │   ├── test_llm_cache_replay.py
│   │   ├── test_adapter_protocol.py
│   │   └── test_cec_pipeline.py
│   ├── paper_alignment/
│   │   ├── test_prediction_score_not_in_prompt.py
│   │   ├── test_summary_not_used_for_training.py
│   │   ├── test_accepted_err_only.py
│   │   ├── test_signed_evidence_masks.py
│   │   └── test_inference_is_llm_free.py
│   └── fixtures/
│       ├── tiny_graph.pt
│       ├── sample_mep.json
│       ├── sample_err_valid.json
│       ├── sample_err_invalid_role.json
│       └── sample_contracts.yaml
├── scripts/
│   ├── run_smoke.sh
│   ├── run_main_table.sh
│   ├── run_ablations.sh
│   ├── check_no_leakage.py
│   ├── check_no_llm_inference.py
│   └── export_results.py
├── notebooks/
│   └── analysis_only/
└── artifacts/
    ├── README.md
    ├── err_cache/
    ├── checkpoints/
    ├── metrics/
    └── tables/
```

关键约束：

```text
src/gread_core/llm/ 只能被 training/stage2_generate_err.py 调用。
src/gread_core/inference/ 不允许 import gread_core.llm。
src/gread_core/models/ 不允许 import openai、anthropic、requests、httpx 等在线 LLM 依赖。
```

---

# 5. 研究方案到代码的一一对应表

| 研究模块                               | 代码模块                                                                                                  | 核心类 / 函数                                                           | 必须测试                                                                | 验收标准                                                                                |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Score-blind MEP                    | `schemas/evidence.py`, `evidence/mep.py`                                                              | `CalibrationChannel`, `ReasoningChannel`, `MinimalEvidencePackage` | `test_mep_score_blind.py`, `test_prediction_score_not_in_prompt.py` | `prediction_score` 只能存在于 calibration channel，不能出现在 prompt、ERR、training labels       |
| Detector-Evidence Adapter Protocol | `adapters/base.py`, `bwgnn_adapter.py`, `caregnn_adapter.py`, `pyg_gnn_adapter.py`, `tree_adapter.py` | `EvidenceAdapter.extract(node_ids)`                                | `test_adapter_protocol.py`                                          | 每个 adapter 输出 `generic + detector_native + counter` 三类 evidence                     |
| Trace selection                    | `tracing/selector.py`, `diversity.py`                                                                 | `TraceSelector.select()`                                           | bucket coverage test                                                | 三桶采样 + bucket 内 evidence diversity；conflict bucket 只做 experimental flag             |
| ERR schema                         | `schemas/err.py`                                                                                      | `EvidenceRationaleRecord`                                          | `test_err_schema.py`                                                | risk type、supporting、counter evidence 必须结构化；summary 不参与训练                           |
| Evidence Contract Verifier         | `verification/verifier.py`, `contract.py`                                                             | `EvidenceContractVerifier.verify()`                                | `test_contract_verifier.py`                                         | schema、availability、role、contract、score-blindness、label compatibility 全部通过才 `a_v=1` |
| Offline LLM teacher                | `llm/teacher.py`, `prompt_builder.py`, `cache.py`                                                     | `LLMTeacher.generate_err()`                                        | `test_llm_cache_replay.py`                                          | LLM 只在 Stage 2 离线调用；输出必须可缓存、可重放                                                     |
| Evidence-conditioned reasoner      | `models/reasoner.py`, `evidence_encoder.py`                                                           | `GReaDReasoner.forward()`                                          | shape / gradient tests                                              | heads 输入 `[z_v; g_v]`；输出 score、risk type、positive mask、negative mask                |
| Evidence-gated residual readout    | `models/residual_readout.py`                                                                          | `EvidenceGatedResidualReadout`                                     | CEC integration test                                                | final logit = base logit + `rho * residual`；`rho` config 控制                         |
| Training protocol                  | `training/stage1_*`, `stage2_*`, `stage3_*`                                                           | CLI + trainer                                                      | `test_stage1_stage2_stage3_tiny.py`                                 | Stage 1 warm-up、Stage 2 ERR generation、Stage 3 distillation 分离                      |
| Loss                               | `losses/reasoning.py`                                                                                 | `ReasoningDistillationLoss`                                        | `test_loss_masking.py`                                              | 只有 `a_v=1` 的样本参与 type/evidence loss                                                 |
| LLM-free inference                 | `inference/predictor.py`                                                                              | `GReaDInferencePipeline`                                           | `test_no_llm_inference.py`                                          | 推理路径无 LLM client import，无网络调用                                                       |
| Tri-CEC                            | `evaluation/cec.py`                                                                                   | `compute_cec_score/type/evidence`                                  | `test_cec_pipeline.py`                                              | 支持 weakening supporting evidence 后评估 score/type/evidence 响应                         |
| Non-redundancy                     | `evaluation/non_redundancy.py`                                                                        | `fit_score_only_vs_score_plus_reasoning()`                         | regression test                                                     | 输出 `Y~P`, `Y~P+T`, `Y~P+T+M` 的 AUC/AUPRC 对比                                         |
| Ablation matrix                    | `evaluation/ablations.py` + configs                                                                   | `run_ablation_suite()`                                             | config validation                                                   | schema-only verifier、score-visible MEP、parallel heads 等可复现                          |

---

# 6. 核心数据结构设计

## 6.1 MEP schema

```python
# src/gread_core/schemas/evidence.py

from typing import Literal
from pydantic import BaseModel, Field


UncertaintyLevel = Literal["low", "medium", "high"]
EvidenceStrength = Literal["weak", "moderate", "strong", "unavailable"]


class CalibrationChannel(BaseModel):
    prediction_score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)


class ReasoningChannel(BaseModel):
    uncertainty_level: UncertaintyLevel
    degree_level: str
    neighbor_consistency: str
    feature_neighbor_discrepancy: str
    detector_signal: str
    detector_signal_strength: EvidenceStrength
    counter_signal: str
    allowed_support_ids: list[str]
    allowed_counter_ids: list[str]


class MinimalEvidencePackage(BaseModel):
    node_id: str
    detector_name: str
    calibration: CalibrationChannel
    reasoning: ReasoningChannel

    def to_teacher_payload(self) -> dict:
        """Return score-blind payload for LLM teacher."""
        return {
            "node_id": self.node_id,
            "detector_name": self.detector_name,
            "reasoning": self.reasoning.model_dump(),
        }
```

Agent 禁止事项：

```text
禁止把 calibration.prediction_score 放入 to_teacher_payload。
禁止把 prediction_score 添加进 allowed_support_ids。
禁止 LLM prompt 引用 fraud score。
```

---

## 6.2 ERR schema

```python
# src/gread_core/schemas/err.py

from typing import Literal
from pydantic import BaseModel, Field


RiskType = Literal[
    "structural_discrepancy",
    "camouflage_neighbor",
    "spectral_anomaly",
    "feature_structure_conflict",
    "relation_or_burst_anomaly",
    "weak_or_uncertain_evidence",
]


class EvidenceRationaleRecord(BaseModel):
    risk_type: RiskType
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    summary: str

    def training_targets(self) -> dict:
        """Only structured fields are allowed for training."""
        return {
            "risk_type": self.risk_type,
            "supporting_evidence": self.supporting_evidence,
            "counter_evidence": self.counter_evidence,
        }
```

Agent 禁止事项：

```text
summary 只能用于日志和人工审计。
summary 不能进入 loss。
summary 不能进入 evidence mask label。
```

---

# 7. Evidence Contract Verifier 的实现合同

Verifier 是整套代码的第一优先级。它不能是“看起来能 parse JSON”的弱校验器，而必须是论文中的 Evidence Contract Verifier。

## 7.1 合同配置

```yaml
# configs/contracts/gread_v1.yaml

risk_types:
  spectral_anomaly:
    required_any:
      - field: detector_signal
        values:
          - high_frequency_response_high
          - spectral_energy_shift_high
          - bandpass_response_high
    optional:
      - field: neighbor_consistency
        values: [low, medium]
      - field: detector_signal_strength
        values: [moderate, strong]
    forbidden:
      - field: detector_signal
        values: [unavailable]
      - field: detector_signal_strength
        values: [weak]

  camouflage_neighbor:
    required_any:
      - field: neighbor_consistency
        values: [low]
      - field: detector_signal
        values:
          - camouflage_neighbor_filter_high
          - neighbor_selection_disagreement_high
    optional:
      - field: degree_level
        values: [high, burst]
      - field: feature_neighbor_discrepancy
        values: [high]
    forbidden:
      - field: neighbor_consistency
        values: [high]

  feature_structure_conflict:
    required_any:
      - field: feature_neighbor_discrepancy
        values: [high]
    optional:
      - field: neighbor_consistency
        values: [low, medium]
    forbidden:
      - field: feature_neighbor_discrepancy
        values: [low, unavailable]

  weak_or_uncertain_evidence:
    required_any:
      - field: uncertainty_level
        values: [high]
      - field: detector_signal
        values: [unavailable]
    optional:
      - field: counter_signal
        values: [benign_neighbor_signal_high, benign_neighbor_signal_medium]
    forbidden: []

role_rules:
  forbidden_support_ids:
    - prediction_score
    - counter_signal
  forbidden_counter_ids:
    - prediction_score
  uncertainty_cannot_support_strong_risk_alone: true

label_compatibility:
  enabled: true
  fraud_label: 1
  benign_label: 0
  fraud_forbidden_risk_types:
    - weak_or_uncertain_evidence
  benign_forbidden_risk_types:
    - camouflage_neighbor
    - spectral_anomaly
    - relation_or_burst_anomaly
```

## 7.2 Verifier 类接口

```python
# src/gread_core/verification/verifier.py

from dataclasses import dataclass

from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.schemas.err import EvidenceRationaleRecord


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reasons: list[str]


class EvidenceContractVerifier:
    def __init__(self, contract_config: dict):
        self.contract_config = contract_config

    def verify(
        self,
        err: EvidenceRationaleRecord,
        mep: MinimalEvidencePackage,
        label: int | None,
    ) -> VerificationResult:
        checks = [
            self._schema_valid(err),
            self._evidence_available(err, mep),
            self._role_consistent(err, mep),
            self._risk_contract_satisfied(err, mep),
            self._score_blind(err),
            self._label_compatible(err, label),
        ]

        reasons = [reason for ok, reason in checks if not ok]
        return VerificationResult(accepted=len(reasons) == 0, reasons=reasons)
```

验收测试必须覆盖：

```text
1. counter_signal 出现在 supporting_evidence → reject
2. prediction_score 出现在任何 evidence list → reject
3. detector_signal=unavailable 但 risk_type=spectral_anomaly → reject
4. y=0 但 risk_type=spectral_anomaly 且无强证据 → reject
5. valid spectral_anomaly contract → accept
6. summary 改写不会影响 accepted 结果
```

---

# 8. Detector-Evidence Adapter Protocol

所有 detector adapter 必须继承同一个抽象类：

```python
# src/gread_core/adapters/base.py

from abc import ABC, abstractmethod
from gread_core.schemas.evidence import MinimalEvidencePackage


class EvidenceAdapter(ABC):
    detector_name: str

    @abstractmethod
    def extract(self, node_ids: list[int]) -> list[MinimalEvidencePackage]:
        """Extract score-blind detector-native MEPs for given nodes."""

    @abstractmethod
    def supports_detector_signal(self) -> bool:
        """Return whether this detector exposes detector-native evidence."""
```

## 8.1 BWGNN adapter

输出：

```text
generic:
  degree_level
  neighbor_consistency
  feature_neighbor_discrepancy
  uncertainty_level

detector_native:
  detector_signal = high_frequency_response_high / bandpass_response_high / neutral / unavailable
  detector_signal_strength = weak / moderate / strong

counter:
  counter_signal = benign_neighbor_signal_low / medium / high
```

BWGNN 代码可以参考官方实现，但在本项目中建议做轻量 PyTorch/PyG 风格封装；官方仓库本身仍是重要 baseline 与 sanity check 来源。([GitHub][12])

## 8.2 CARE-GNN adapter

输出：

```text
detector_signal:
  camouflage_neighbor_filter_high
  neighbor_selection_disagreement_high
  relation_aware_camouflage_signal
```

CARE-GNN 官方实现强调 label-aware similarity、neighbor selector、relation-aware aggregator，因此 adapter 应优先暴露这些 camouflage-native 信号。([GitHub][13])

## 8.3 GCN / GAT / GraphSAGE adapter

输出：

```text
detector_signal:
  embedding_neighbor_discrepancy_high
  attention_concentration_high
  message_disagreement_high
```

## 8.4 Tree + neighborhood aggregation adapter

输出：

```text
detector_signal:
  feature_importance_risk_high
  neighborhood_aggregation_discrepancy_high
```

这样可以应对 GADBench 强调的传统模型强基线压力，而不是只做 GNN-only demo。

---

# 9. Student Reasoner 设计

## 9.1 Evidence encoder

```python
class EvidenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * NUM_EVIDENCE_SLOTS, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

    def forward(self, evidence_token_ids: torch.LongTensor) -> torch.Tensor:
        # evidence_token_ids: [batch, num_slots]
        x = self.embedding(evidence_token_ids)
        x = x.flatten(start_dim=1)
        return self.proj(x)
```

## 9.2 Reasoner forward contract

```python
class GReaDReasoner(nn.Module):
    def __init__(self, base_detector, evidence_encoder, hidden_dim, num_risk_types, num_evidence_slots):
        super().__init__()
        self.base_detector = base_detector
        self.evidence_encoder = evidence_encoder
        self.type_head = nn.Linear(hidden_dim + 128, num_risk_types)
        self.pos_evidence_head = nn.Linear(hidden_dim + 128, num_evidence_slots)
        self.neg_evidence_head = nn.Linear(hidden_dim + 128, num_evidence_slots)
        self.residual_readout = EvidenceGatedResidualReadout(hidden_dim, 128)

    def forward(self, batch):
        base_logit, z = self.base_detector.forward_with_embedding(batch.graph)
        g = self.evidence_encoder(batch.evidence_token_ids)
        h = torch.cat([z, g], dim=-1)

        type_logits = self.type_head(h)
        pos_mask_logits = self.pos_evidence_head(h)
        neg_mask_logits = self.neg_evidence_head(h)

        residual_logit = self.residual_readout(
            z=z,
            evidence_embedding=g,
            pos_mask_logits=pos_mask_logits,
            neg_mask_logits=neg_mask_logits,
        )

        final_logit = base_logit + batch.config.residual_rho * residual_logit

        return {
            "base_logit": base_logit,
            "final_logit": final_logit,
            "type_logits": type_logits,
            "pos_mask_logits": pos_mask_logits,
            "neg_mask_logits": neg_mask_logits,
        }
```

注意：final score 可以受 evidence residual 影响，但不能让 evidence residual 盖过 detector 主干。`residual_rho` 默认建议：

```yaml
residual_rho: 0.1
```

---

# 10. Loss 实现

```python
def reasoning_distillation_loss(outputs, batch, lambda_reason: float):
    sup_loss = F.binary_cross_entropy_with_logits(
        outputs["final_logit"],
        batch.labels.float(),
    )

    accepted = batch.accepted_mask.bool()

    if accepted.sum() == 0:
        return {
            "loss": sup_loss,
            "sup_loss": sup_loss.detach(),
            "type_loss": torch.tensor(0.0, device=sup_loss.device),
            "evidence_loss": torch.tensor(0.0, device=sup_loss.device),
        }

    type_loss = F.cross_entropy(
        outputs["type_logits"][accepted],
        batch.risk_type_targets[accepted],
    )

    pos_loss = F.binary_cross_entropy_with_logits(
        outputs["pos_mask_logits"][accepted],
        batch.pos_evidence_targets[accepted].float(),
    )

    neg_loss = F.binary_cross_entropy_with_logits(
        outputs["neg_mask_logits"][accepted],
        batch.neg_evidence_targets[accepted].float(),
    )

    evidence_loss = pos_loss + neg_loss
    loss = sup_loss + lambda_reason * (type_loss + evidence_loss)

    return {
        "loss": loss,
        "sup_loss": sup_loss.detach(),
        "type_loss": type_loss.detach(),
        "evidence_loss": evidence_loss.detach(),
    }
```

Agent 禁止事项：

```text
不能让 rejected ERR 样本进入 type/evidence loss。
不能把 accepted_mask 当 soft weight。
不能训练 summary。
不能新增 DHEF / CER 为默认主方法；这些只能放 experimental/ 并默认关闭。
```

---

# 11. Training pipeline

## Stage 1: base detector warm-up

CLI：

```bash
python -m gread_core.cli.train_detector \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --seed 1
```

输出：

```text
artifacts/checkpoints/stage1_detector.pt
artifacts/metrics/stage1_detector_metrics.json
artifacts/mep/stage1_mep_train.jsonl
```

验收：

```text
1. 能在 tiny graph 上 CPU 跑通。
2. 能保存 base score、embedding、uncertainty。
3. 不能生成 ERR。
4. 不能调用 LLM。
```

## Stage 2: offline MEP → LLM ERR → ECV

CLI：

```bash
python -m gread_core.cli.generate_err \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --checkpoint artifacts/checkpoints/stage1_detector.pt \
  --output artifacts/err_cache/yelp_bwgnn_seed1.jsonl
```

流程：

```text
1. TraceSelector 选 trace nodes。
2. Adapter 抽取 MEP。
3. PromptBuilder 只读取 MEP.reasoning。
4. LLMTeacher 生成 ERR。
5. EvidenceContractVerifier 输出 accepted/rejected。
6. 只保存 accepted ERR 的 training targets。
```

验收：

```text
1. prompt 中 grep 不到 prediction_score。
2. accepted ERR 必须有 verifier reasons 日志。
3. LLM 输出必须 cache，以便 replay。
4. 同一 prompt hash 重跑不能再次调用 LLM。
```

## Stage 3: train evidence-conditioned reasoner

CLI：

```bash
python -m gread_core.cli.train_reasoner \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --detector-checkpoint artifacts/checkpoints/stage1_detector.pt \
  --err-cache artifacts/err_cache/yelp_bwgnn_seed1.jsonl
```

验收：

```text
1. 使用 L = L_sup + λ a_v (L_type + L_evidence)。
2. 输出 fraud score、risk type、positive evidence mask、negative evidence mask。
3. 推理时无需 LLM。
4. 保存完整 config、seed、git commit、dataset split hash。
```

---

# 12. Evaluation pipeline

CLI：

```bash
python -m gread_core.cli.evaluate \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --checkpoint artifacts/checkpoints/stage3_reasoner.pt \
  --err-cache artifacts/err_cache/yelp_bwgnn_seed1.jsonl
```

输出：

```text
artifacts/metrics/detection.json
artifacts/metrics/reasoning.json
artifacts/metrics/cec.json
artifacts/metrics/non_redundancy.json
artifacts/tables/main_results.csv
artifacts/tables/ablation_results.csv
```

## 12.1 Detection metrics

```text
ROC-AUC
AUPRC
Recall@K
Precision@K
F1
```

## 12.2 Reasoning metrics

```text
Verifier acceptance rate
Contract violation rate
Risk-type agreement
Evidence positive-mask F1
Evidence negative-mask F1
Evidence sparsity
Template validity
```

## 12.3 Tri-CEC

```python
def compute_tri_cec(model, batch, perturb_fn):
    full = model(batch)
    weak_batch = perturb_fn(batch)
    weak = model(weak_batch)

    cec_score = (
        torch.sigmoid(full["final_logit"]) -
        torch.sigmoid(weak["final_logit"])
        > 0
    ).float().mean()

    true_type_prob_full = full["type_logits"].softmax(dim=-1).gather(
        1, batch.risk_type_targets[:, None]
    )
    true_type_prob_weak = weak["type_logits"].softmax(dim=-1).gather(
        1, batch.risk_type_targets[:, None]
    )
    cec_type = (true_type_prob_full - true_type_prob_weak > 0).float().mean()

    pos_full = torch.sigmoid(full["pos_mask_logits"])
    pos_weak = torch.sigmoid(weak["pos_mask_logits"])
    cec_evidence = ((pos_full - pos_weak) * batch.pos_evidence_targets > 0).float().mean()

    return {
        "cec_score": cec_score.item(),
        "cec_type": cec_type.item(),
        "cec_evidence": cec_evidence.item(),
    }
```

## 12.4 Non-redundancy test

必须实现三组模型：

```text
Y ~ P
Y ~ P + T
Y ~ P + T + M
```

其中：

```text
P = base prediction score
T = risk type probabilities
M = signed evidence mask probabilities
```

输出：

```json
{
  "auc_score_only": 0.842,
  "auc_score_plus_type": 0.855,
  "auc_score_plus_type_evidence": 0.862,
  "auprc_score_only": 0.412,
  "auprc_score_plus_type": 0.435,
  "auprc_score_plus_type_evidence": 0.451
}
```

验收标准：

```text
如果 score_plus_type_evidence 没有超过 score_only，论文中不能声称 reasoning outputs provide non-redundant information。
```

---

# 13. Agent 指令文件：AGENTS.md

这个文件是给 Codex、GitHub Agent、Cursor 等通用 coding agent 的根规则。Codex 官方文档明确说明项目内 `AGENTS.md` 会被层级读取，所以这个文件应放在 repo root。([OpenAI 开发者][5])

````markdown
# AGENTS.md

## Project Identity

This repository implements GReaD-Core:
Contract-Verified Score-Blind Evidence Distillation for LLM-Free Graph Fraud Reasoning.

The implementation must strictly follow the paper design:
Score-blind MEP -> Detector-Evidence Adapter -> Evidence Contract Verifier -> Offline LLM ERR -> Evidence-conditioned residual reasoner -> LLM-free inference -> tri-CEC and non-redundancy evaluation.

## Non-Negotiable Research Constraints

1. `prediction_score` is calibration-only.
   - It may appear in `CalibrationChannel`.
   - It must never appear in LLM prompts.
   - It must never appear in supporting_evidence or counter_evidence.
   - It must never be used as an evidence target.

2. LLM is training-offline only.
   - LLM code must stay inside `src/gread_core/llm`.
   - Inference code must not import `gread_core.llm`.
   - Models must not import OpenAI, Anthropic, HTTP clients, or network code.

3. Verifier must be hard and deterministic.
   - No LLM-as-judge in the main verifier.
   - No learned verifier in the main method.
   - Accepted ERR requires schema, availability, role consistency, contract consistency, score-blindness, and label compatibility.

4. Training objective must stay aligned:
   `L = L_sup + lambda * a_v * (L_type + L_evidence)`.
   - Rejected ERR samples must not contribute to type/evidence losses.
   - `summary` must not be used for training.
   - DHEF, CER, ECB are experimental only and disabled by default.

5. Inference output:
   - fraud_score
   - risk_type
   - supporting_evidence
   - counter_evidence
   - deterministic template explanation
   - no LLM call

## Engineering Rules

- Use Python 3.11+.
- Use PyTorch as the core deep learning framework.
- Prefer PyTorch Geometric-style data objects internally.
- Use Pydantic for schemas and validation.
- Use Hydra/OmegaConf or plain YAML config loading; no hidden constants.
- Use ruff, mypy, pytest, and pre-commit.
- Every public class must have type hints.
- Every research contract must have tests.
- Every PR must include:
  - tests added or updated
  - config added or updated
  - documentation updated when behavior changes
  - smoke command result

## Required Commands Before Finishing Any Task

Run:

```bash
ruff check .
mypy src
pytest tests/unit
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py
````

For training-related changes, also run:

```bash
bash scripts/run_smoke.sh
```

## Forbidden Patterns

* Do not add online LLM calls to inference.
* Do not add `prediction_score` to teacher prompts.
* Do not use free-form rationale text as a training label.
* Do not silently change risk taxonomy.
* Do not hard-code dataset-specific constants in model code.
* Do not add large dependencies without updating `pyproject.toml`, README, and justification.
* Do not mix multiple unrelated features in one PR.

## Implementation Style

* Small modules.
* Clear interfaces.
* Config-driven behavior.
* Deterministic tests.
* Minimal magic.
* Fail fast with explicit error messages.

````

---

# 14. Claude Code Skill：`.claude/skills/gread-core-implementer/SKILL.md`

Claude 官方 Skills 是 `SKILL.md` 文件系统 artifact，可以被自动发现并按上下文调用。:contentReference[oaicite:18]{index=18}

```markdown
---
name: gread-core-implementer
description: Implement and verify GReaD-Core research code while preserving strict paper alignment.
---

# GReaD-Core Implementer Skill

Use this skill when implementing, refactoring, testing, or reviewing code in the GReaD-Core repository.

## Research Contract

GReaD-Core consists of:

1. Score-blind Minimal Evidence Package
2. Detector-Evidence Adapter Protocol
3. Evidence Rationale Record
4. Evidence Contract Verifier
5. Offline LLM teacher generation
6. Evidence-conditioned residual student reasoner
7. Two-stage/three-step training protocol
8. LLM-free inference
9. Tri-CEC and non-redundancy evaluation

## Mandatory Alignment Checklist

Before coding, identify which paper component is being modified:

- MEP
- Adapter
- Trace selection
- ERR
- Verifier
- Teacher
- Student reasoner
- Loss
- Training
- Evaluation
- Inference

If the requested change does not map to one of these components, ask whether it belongs in `experimental/`.

## Default Implementation Procedure

1. Read the relevant spec under `specs/`.
2. Read the relevant config under `configs/`.
3. Locate or create the module under `src/gread_core/`.
4. Create or update unit tests first.
5. Implement the smallest passing change.
6. Run unit tests.
7. Run paper alignment tests.
8. Update docs if behavior changed.

## Non-Negotiable Rules

- Never expose `prediction_score` to the LLM teacher.
- Never use ERR summary as training signal.
- Never allow rejected ERR into reasoning loss.
- Never import LLM code in inference.
- Never modify the risk taxonomy without updating:
  - `schemas/risk_taxonomy.py`
  - `configs/contracts/gread_v1.yaml`
  - verifier tests
  - README
  - specs

## Required Validation Commands

```bash
ruff check .
mypy src
pytest tests/unit
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py
````

For model or training changes:

```bash
bash scripts/run_smoke.sh
```

## Code Review Lens

Reject changes that:

* add untested verifier behavior
* create hidden score leakage
* make inference depend on LLM
* add unconfigured hyperparameters
* mix baseline, ablation, and main method logic
* produce non-reproducible experiment outputs

````

---

# 15. Spec Kit 工作流提示词

Spec Kit 官方流程建议从 spec、plan、tasks 到 implement，把 AI agent 当 literal-minded pair programmer，而不是搜索引擎。:contentReference[oaicite:19]{index=19}

## 15.1 Constitution prompt

用于 `/speckit.constitution`：

```text
Create the governing principles for GReaD-Core, a PyTorch research framework for contract-verified score-blind evidence distillation in graph fraud detection.

The constitution must enforce:
1. strict paper-to-code alignment;
2. score-blind MEP by construction;
3. deterministic Evidence Contract Verifier;
4. offline-only LLM teacher;
5. LLM-free inference;
6. PyTorch/PyG-style modular architecture;
7. reproducible experiments with configs, seeds, and cached teacher outputs;
8. pytest/ruff/mypy/CI quality gates;
9. no free-form rationale text in training;
10. all experimental extensions disabled by default.
````

## 15.2 Specify prompt

用于 `/speckit.specify`：

```text
Specify the GReaD-Core implementation.

Functional requirements:
- Load graph fraud datasets.
- Train a base detector.
- Extract score-blind Minimal Evidence Packages through detector-specific adapters.
- Select trace nodes using 3-bucket plus evidence diversity sampling.
- Generate Evidence Rationale Records offline with an LLM teacher.
- Verify ERRs using Evidence Contract Verifier.
- Train an evidence-conditioned residual reasoner with signed evidence masks.
- Evaluate detection, reasoning quality, tri-CEC, and non-redundancy.
- Run inference without LLM.

Non-functional requirements:
- Python 3.11+, PyTorch, PyG-style data.
- Pydantic schemas.
- Config-driven experiments.
- Deterministic tests and caches.
- No score leakage.
- No LLM import in inference.
```

## 15.3 Plan prompt

用于 `/speckit.plan`：

```text
Create a technical implementation plan for GReaD-Core using PyTorch, PyTorch Geometric-style graph data, Pydantic schemas, Hydra/OmegaConf YAML configs, pytest, ruff, mypy, and GitHub Actions.

The plan must split implementation into:
1. repository harness and CI;
2. schemas and risk taxonomy;
3. MEP and leakage guards;
4. detector adapters;
5. trace selection;
6. LLM teacher and cache;
7. Evidence Contract Verifier;
8. student reasoner;
9. losses and training stages;
10. evaluation metrics;
11. inference pipeline;
12. ablation runner.
```

## 15.4 Tasks prompt

用于 `/speckit.tasks`：

```text
Break the GReaD-Core implementation plan into small reviewable GitHub issues.

Each task must include:
- target files;
- paper component mapping;
- acceptance criteria;
- tests to add;
- commands to run;
- forbidden shortcuts;
- expected artifacts.

No task may modify more than one major paper component unless it is a pure integration task.
```

---

# 16. GitHub Issues 切片

## Epic 0: Harness and repository bootstrap

**目标：** 先建立“AI 不能跑偏”的环境。

Tasks：

```text
0.1 Create pyproject, ruff, mypy, pytest, pre-commit.
0.2 Create AGENTS.md and CLAUDE.md.
0.3 Create specs directory and initial constitution.
0.4 Create CI workflow.
0.5 Create scripts/check_no_leakage.py and scripts/check_no_llm_inference.py.
```

Definition of Done：

```text
ruff check .
mypy src
pytest
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py
```

## Epic 1: Schemas and taxonomy

Tasks：

```text
1.1 Implement evidence schemas.
1.2 Implement ERR schema.
1.3 Implement risk taxonomy.
1.4 Implement config validation.
```

Acceptance：

```text
Invalid evidence ids fail validation.
Unknown risk type fails validation.
ERR summary is not included in training_targets().
```

## Epic 2: Score-blind MEP and leakage guard

Tasks：

```text
2.1 Implement MEP builder.
2.2 Implement CalibrationChannel and ReasoningChannel.
2.3 Implement teacher payload method.
2.4 Implement leakage guard tests.
```

Acceptance：

```text
grep artifacts/prompts -r prediction_score returns nothing.
test_prediction_score_not_in_prompt passes.
```

## Epic 3: Detector adapters

Tasks：

```text
3.1 Implement EvidenceAdapter abstract base.
3.2 Implement generic evidence signals.
3.3 Implement BWGNN adapter.
3.4 Implement CARE-GNN adapter.
3.5 Implement GCN/GAT/GraphSAGE adapter.
3.6 Implement tree-neighbor adapter.
```

Acceptance：

```text
Every adapter returns generic, detector_native, and counter evidence.
Adapter outputs are valid MEPs.
```

## Epic 4: Verifier

Tasks：

```text
4.1 Implement schema verifier.
4.2 Implement availability verifier.
4.3 Implement role consistency verifier.
4.4 Implement contract verifier from YAML.
4.5 Implement score-blindness verifier.
4.6 Implement label compatibility verifier.
4.7 Implement VerificationResult logging.
```

Acceptance：

```text
All invalid ERR fixtures are rejected for the right reason.
All valid ERR fixtures are accepted.
Verifier is deterministic.
```

## Epic 5: LLM teacher

Tasks：

```text
5.1 Implement prompt template.
5.2 Implement prompt builder.
5.3 Implement LLM client interface.
5.4 Implement JSON parser and retry policy.
5.5 Implement cache keyed by prompt hash.
5.6 Implement replay mode.
```

Acceptance：

```text
Offline cache can replay ERR generation without network.
Teacher payload is score-blind.
```

## Epic 6: Student reasoner

Tasks：

```text
6.1 Implement EvidenceEncoder.
6.2 Implement RiskTypeHead.
6.3 Implement SignedEvidenceHead.
6.4 Implement EvidenceGatedResidualReadout.
6.5 Implement GReaDReasoner.
```

Acceptance：

```text
Forward pass returns base_logit, final_logit, type_logits, pos_mask_logits, neg_mask_logits.
Changing evidence tokens changes type/evidence outputs.
rho=0 recovers base detector logits.
```

## Epic 7: Training pipeline

Tasks：

```text
7.1 Implement Stage 1 detector training.
7.2 Implement Stage 2 ERR generation.
7.3 Implement Stage 3 reasoner distillation.
7.4 Implement checkpointing and logging.
7.5 Implement tiny graph smoke training.
```

Acceptance：

```text
bash scripts/run_smoke.sh finishes on CPU.
Reasoning loss is zero when accepted_mask has no positives.
```

## Epic 8: Evaluation

Tasks：

```text
8.1 Detection metrics.
8.2 Reasoning metrics.
8.3 Tri-CEC.
8.4 Non-redundancy test.
8.5 Ablation runner.
8.6 Export tables.
```

Acceptance：

```text
Evaluation produces JSON metrics and CSV tables.
Ablation configs are reproducible.
```

## Epic 9: Reproducibility package

Tasks：

```text
9.1 Add README commands.
9.2 Add experiment manifest.
9.3 Add result export script.
9.4 Add artifact structure docs.
9.5 Add paper table generator.
```

Acceptance：

```text
A new researcher can reproduce smoke, main, and ablation tables from README.
```

---

# 17. CI / Quality gates

## 17.1 GitHub Actions

```yaml
name: ci

on:
  pull_request:
  push:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: |
          python -m pip install -U pip
          pip install -e ".[dev]"
      - name: Ruff
        run: ruff check .
      - name: Mypy
        run: mypy src
      - name: Unit tests
        run: pytest tests/unit
      - name: Paper alignment tests
        run: pytest tests/paper_alignment
      - name: No score leakage
        run: python scripts/check_no_leakage.py
      - name: No LLM inference
        run: python scripts/check_no_llm_inference.py
```

## 17.2 Smoke workflow

```yaml
name: smoke

on:
  pull_request:
    paths:
      - "src/gread_core/models/**"
      - "src/gread_core/training/**"
      - "src/gread_core/evaluation/**"
      - "configs/**"

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: bash scripts/run_smoke.sh
```

## 17.3 Paper alignment workflow

```yaml
name: paper-alignment

on:
  pull_request:

jobs:
  alignment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/paper_alignment
      - run: python scripts/check_no_leakage.py
      - run: python scripts/check_no_llm_inference.py
```

---

# 18. Paper alignment tests 的具体规则

## 18.1 No score leakage

```python
def test_prediction_score_not_in_teacher_payload(sample_mep):
    payload = sample_mep.to_teacher_payload()
    assert "prediction_score" not in str(payload)
```

脚本级检查：

```python
# scripts/check_no_leakage.py

from pathlib import Path

FORBIDDEN_DIRS = [
    "src/gread_core/llm/templates",
    "configs/prompts",
]

for directory in FORBIDDEN_DIRS:
    for path in Path(directory).rglob("*"):
        if path.is_file() and "prediction_score" in path.read_text():
            raise SystemExit(f"Score leakage detected in {path}")
```

## 18.2 No LLM inference

```python
# scripts/check_no_llm_inference.py

from pathlib import Path

FORBIDDEN_IMPORTS = [
    "openai",
    "anthropic",
    "gread_core.llm",
    "requests",
    "httpx",
]

for path in Path("src/gread_core/inference").rglob("*.py"):
    text = path.read_text()
    for token in FORBIDDEN_IMPORTS:
        if token in text:
            raise SystemExit(f"LLM/network import detected in inference: {path}: {token}")
```

## 18.3 Accepted ERR only

```python
def test_rejected_err_has_zero_reasoning_loss(reasoning_loss, sample_batch_rejected):
    result = reasoning_loss(sample_batch_rejected)
    assert result["type_loss"].item() == 0.0
    assert result["evidence_loss"].item() == 0.0
```

## 18.4 Summary not used

```python
def test_err_summary_not_in_training_targets(sample_err):
    targets = sample_err.training_targets()
    assert "summary" not in targets
```

---

# 19. 配置系统建议

## 19.1 Default config

```yaml
project:
  name: gread-core
  seed: 1
  output_dir: artifacts

method:
  score_blind: true
  lambda_reason: 0.5
  residual_rho: 0.1
  signed_evidence_masks: true
  use_llm_at_inference: false

risk_taxonomy:
  types:
    - structural_discrepancy
    - camouflage_neighbor
    - spectral_anomaly
    - feature_structure_conflict
    - relation_or_burst_anomaly
    - weak_or_uncertain_evidence

training:
  stage1:
    epochs: 100
    lr: 0.001
    weight_decay: 0.0001
  stage3:
    epochs: 100
    lr: 0.001
    warmup_epochs: 5

trace_selection:
  total_budget: 1000
  buckets:
    uncertain: 0.333
    high_conf_fraud: 0.333
    high_conf_benign: 0.334
  diversity_sampling: true
  conflict_bucket:
    enabled: false

llm_teacher:
  enabled: true
  mode: offline_cache
  temperature: 0.0
  max_retries: 2
  cache_dir: artifacts/err_cache

verifier:
  contract_path: configs/contracts/gread_v1.yaml
  label_compatibility: true
  schema_only_ablation: false

evaluation:
  detection: true
  reasoning: true
  tri_cec: true
  non_redundancy: true
```

## 19.2 Ablation configs

```yaml
# configs/experiments/ablation_score_visible.yaml

inherits: configs/experiments/main_bwgnn_yelp.yaml

method:
  score_blind: false

paper_warning:
  purpose: "Ablation only. Not main method."
```

```yaml
# configs/experiments/ablation_schema_only_verifier.yaml

inherits: configs/experiments/main_bwgnn_yelp.yaml

verifier:
  schema_only_ablation: true
  label_compatibility: false

paper_warning:
  purpose: "Ablation only. Tests verifier strength."
```

```yaml
# configs/experiments/ablation_parallel_heads.yaml

inherits: configs/experiments/main_bwgnn_yelp.yaml

method:
  residual_rho: 0.0

paper_warning:
  purpose: "Ablation only. Tests disconnected rationale risk."
```

---

# 20. Prompt template 设计

## 20.1 Jinja template

```jinja2
You are a graph fraud detection evidence analyst.

You must generate a structured Evidence Rationale Record.

Rules:
1. Use only the evidence IDs listed in allowed_support_ids and allowed_counter_ids.
2. Do not mention prediction score, fraud score, probability, or model confidence.
3. counter_signal can only be counter_evidence.
4. uncertainty_level alone cannot support a strong fraud risk type.
5. Choose exactly one risk_type from the taxonomy.
6. Output valid JSON only.

Risk taxonomy:
- structural_discrepancy
- camouflage_neighbor
- spectral_anomaly
- feature_structure_conflict
- relation_or_burst_anomaly
- weak_or_uncertain_evidence

Node evidence:
{{ reasoning_payload_json }}

Return JSON:
{
  "risk_type": "...",
  "supporting_evidence": ["..."],
  "counter_evidence": ["..."],
  "summary": "..."
}
```

验收：

```text
1. 模板中不得出现 prediction_score。
2. 模板中不得要求 LLM 估计 fraud probability。
3. LLM 输出必须被 verifier 二次过滤。
```

---

# 21. AI Agent 可直接使用的开发提示词

## 21.1 给 Codex 的任务模板

```text
Read AGENTS.md and specs/003_evidence_contract_verifier.md.

Implement the Evidence Contract Verifier only.

Target files:
- src/gread_core/verification/schema.py
- src/gread_core/verification/role_consistency.py
- src/gread_core/verification/contract.py
- src/gread_core/verification/label_compatibility.py
- src/gread_core/verification/verifier.py
- tests/unit/test_contract_verifier.py
- tests/fixtures/sample_contracts.yaml

Research constraints:
- Hard deterministic verifier only.
- No LLM-as-judge.
- No learned verifier.
- prediction_score must be rejected if cited as evidence.
- counter_signal must be rejected if cited as supporting evidence.
- label compatibility must be configurable and enabled by default.

Before finishing, run:
ruff check .
mypy src
pytest tests/unit/test_contract_verifier.py
pytest tests/paper_alignment
```

## 21.2 给 Claude Code 的任务模板

```text
Use the gread-core-implementer skill.

Implement the score-blind MEP module.

You must:
1. Create Pydantic schemas for CalibrationChannel, ReasoningChannel, and MinimalEvidencePackage.
2. Implement to_teacher_payload() that excludes calibration.
3. Add leakage guard tests.
4. Add fixtures.
5. Update specs/001_score_blind_mep.md.

Forbidden:
- Do not expose prediction_score to teacher payload.
- Do not add prediction_score to allowed_support_ids.
- Do not implement LLM teacher in this task.

Run:
ruff check .
mypy src
pytest tests/unit/test_mep_score_blind.py
pytest tests/paper_alignment/test_prediction_score_not_in_prompt.py
```

## 21.3 给 Code Review Agent 的任务模板

```text
Review this PR for GReaD-Core paper alignment.

Check:
1. Does every code change map to a paper component?
2. Does any prompt, schema, or training target leak prediction_score?
3. Does inference import LLM or network code?
4. Are rejected ERRs excluded from reasoning losses?
5. Is summary excluded from training?
6. Are new constants config-driven?
7. Are tests added for every new contract?
8. Is this change main-method or ablation/experimental?

Reject the PR if any research constraint is violated.
```

---

# 22. PyTorch 研发规范

## 22.1 Model design

```text
1. base detector 和 reasoner 分离。
2. base detector 必须支持 forward_with_embedding。
3. evidence encoder 独立模块。
4. risk head、positive evidence head、negative evidence head 独立模块。
5. residual readout 独立模块，rho 可设为 0 做 ablation。
6. 所有 tensor shape 在 docstring 中写清楚。
```

## 22.2 Tensor shape contract

```text
node embeddings z:              [B, H]
evidence token ids:             [B, K]
evidence embedding g:           [B, E]
base_logit:                     [B]
final_logit:                    [B]
type_logits:                    [B, T]
pos_mask_logits:                [B, K]
neg_mask_logits:                [B, K]
accepted_mask:                  [B]
risk_type_targets:              [B]
pos_evidence_targets:           [B, K]
neg_evidence_targets:           [B, K]
```

## 22.3 Reproducibility

每次实验必须写入：

```json
{
  "experiment_id": "...",
  "git_commit": "...",
  "config_hash": "...",
  "dataset": "...",
  "split_hash": "...",
  "seed": 1,
  "base_detector_checkpoint": "...",
  "err_cache_hash": "...",
  "contract_version": "gread_v1",
  "created_at": "..."
}
```

## 22.4 Device and batching

```text
1. tiny tests 默认 CPU。
2. full experiments 支持 CUDA。
3. LLM generation 不依赖 GPU。
4. MEP/ERR cache 采用 JSONL。
5. 大图训练使用 sampler，但 adapter 输出必须保持 node_id 对齐。
```

---

# 23. 主方法与 experimental 功能边界

为了防止 agent 把 18 条改进意见全部塞进主方法，必须在代码中明确：

```text
mainline:
  - score-blind MEP
  - detector adapter protocol
  - evidence diversity trace selection
  - ERR
  - Evidence Contract Verifier
  - label compatibility
  - evidence-conditioned reasoner
  - evidence-gated residual readout
  - signed evidence masks
  - tri-CEC evaluation
  - non-redundancy evaluation

experimental, disabled by default:
  - DHEF
  - CER as training regularizer
  - evidence-conflict bucket
  - multi-sample LLM self-consistency
  - prototype prompt update
  - adaptive lambda
```

目录建议：

```text
src/gread_core/experimental/
├── dhef.py
├── cer.py
├── conflict_bucket.py
├── prototype_prompting.py
└── adaptive_lambda.py
```

任何 experimental 代码不得被 `configs/experiments/main_*.yaml` 默认启用。

---

# 24. 敏捷开发节奏

## Sprint 0：Harness foundation

交付：

```text
AGENTS.md
CLAUDE.md
Spec Kit constitution
pyproject
CI
pre-commit
basic package structure
```

成功标准：

```text
空仓库质量门禁可运行。
AI Agent 有明确行为规范。
```

## Sprint 1：Schemas + verifier

交付：

```text
MEP schema
ERR schema
Risk taxonomy
Evidence Contract Verifier
Verifier fixtures
Paper alignment tests
```

成功标准：

```text
所有 verifier 攻击样例都能被拒绝。
```

## Sprint 2：Data + adapters

交付：

```text
dataset loaders
generic evidence signals
BWGNN adapter
GCN/GAT adapter
tree adapter
tiny graph fixture
```

成功标准：

```text
每个 adapter 输出合法 MEP。
```

## Sprint 3：LLM teacher offline pipeline

交付：

```text
prompt template
teacher client abstraction
JSON parser
cache/replay
Stage 2 CLI
```

成功标准：

```text
cache replay 不调用网络。
prompt score-blind。
```

## Sprint 4：Student reasoner + training

交付：

```text
EvidenceEncoder
GReaDReasoner
losses
Stage 1/3 trainer
smoke script
```

成功标准：

```text
tiny graph CPU 端到端跑通。
```

## Sprint 5：Evaluation + ablations

交付：

```text
detection metrics
reasoning metrics
tri-CEC
non-redundancy
ablation configs
table exporter
```

成功标准：

```text
能生成主表、消融表、CEC 表。
```

## Sprint 6：Research hardening

交付：

```text
README reproduction
experiment registry
seed stability
config audit
PR review checklist
```

成功标准：

```text
新机器按 README 能跑通 smoke 和至少一个主实验。
```

---

# 25. PR Checklist

每个 PR 必须包含：

````markdown
## Paper Component Mapping

This PR modifies:
- [ ] MEP
- [ ] Adapter
- [ ] Trace selection
- [ ] ERR
- [ ] Verifier
- [ ] LLM teacher
- [ ] Student reasoner
- [ ] Loss
- [ ] Training
- [ ] Evaluation
- [ ] Inference
- [ ] Experimental only

## Research Alignment

- [ ] No prediction_score leakage.
- [ ] No LLM in inference.
- [ ] Rejected ERRs do not contribute to reasoning loss.
- [ ] ERR summary is not used for training.
- [ ] Risk taxonomy unchanged or fully updated.
- [ ] Config updated for new behavior.
- [ ] Tests added.

## Commands Run

```bash
ruff check .
mypy src
pytest tests/unit
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py
````

## Artifacts

* Metrics:
* Logs:
* Screenshots / tables:

````

---

# 26. 最小可运行版本 MVP

MVP 不需要一开始就实现所有 detector。建议最小闭环：

```text
Dataset:
  tiny graph fixture + YelpChi or Amazon

Detector:
  PyG GCN baseline + BWGNN adapter stub

MEP:
  score-blind reasoning channel

Verifier:
  full ECV

Teacher:
  cache replay first, live LLM second

Reasoner:
  evidence-conditioned heads + residual readout

Evaluation:
  ROC-AUC, AUPRC, verifier acceptance, evidence F1, tri-CEC
````

MVP 验收命令：

```bash
bash scripts/run_smoke.sh
python -m gread_core.cli.evaluate \
  --config configs/experiments/smoke_tiny.yaml \
  --checkpoint artifacts/checkpoints/smoke_reasoner.pt
```

MVP 成功后再扩展：

```text
BWGNN full
CARE-GNN adapter
tree-neighbor baseline
GADBench-style datasets
full ablation suite
```

---

# 27. 最终可交付物清单

项目最终应交付：

```text
1. 可安装 Python package
2. 完整 AGENTS.md / CLAUDE.md / Skill
3. specs/ 研究契约文档
4. configs/ 主实验和消融实验
5. src/gread_core/ 主代码
6. tests/ 单元、集成、paper alignment 测试
7. scripts/ smoke、main、ablation、export
8. artifacts/err_cache 示例
9. README 复现实验命令
10. paper tables 自动导出
```

论文复现实验命令最终应长这样：

```bash
# Stage 1
python -m gread_core.cli.train_detector \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --seed 1

# Stage 2
python -m gread_core.cli.generate_err \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --seed 1

# Stage 3
python -m gread_core.cli.train_reasoner \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --seed 1

# Evaluation
python -m gread_core.cli.evaluate \
  --config configs/experiments/main_bwgnn_yelp.yaml \
  --seed 1

# Ablations
bash scripts/run_ablations.sh

# Export paper tables
python scripts/export_results.py \
  --metrics-dir artifacts/metrics \
  --output-dir artifacts/tables
```

---

# 28. 一句话执行原则

**不要让 AI Agent “实现一个图欺诈检测想法”；要让它在 harness 中逐个实现经过论文契约约束的模块。**

对 GReaD-Core 来说，最重要的 harness 不是 CI 本身，而是这五条不可破坏的研究契约：

```text
1. prediction_score 永不进入 LLM reasoning。
2. verifier 必须是 hard, deterministic, contract-based。
3. ERR summary 永不进入训练。
4. 推理路径永不调用 LLM。
5. 所有输出必须能回到 MEP evidence slots 和 paper metrics。
```

只要这五条被 `AGENTS.md + tests + CI + PR checklist + Skill` 同时锁住，Claude Code、Codex 这类 agent 就可以安全地高速实现，而不会把研究方案写偏。

[1]: https://openai.com/index/harness-engineering/ "Harness engineering: leveraging Codex in an agent-first world | OpenAI"
[2]: https://github.blog/ai-and-ml/generative-ai/spec-driven-development-with-ai-get-started-with-a-new-open-source-toolkit/ "Spec-driven development with AI: Get started with a new open source toolkit - The GitHub Blog"
[3]: https://github.com/github/spec-kit "GitHub - github/spec-kit:  Toolkit to help you get started with Spec-Driven Development · GitHub"
[4]: https://github.com/openai/codex "GitHub - openai/codex: Lightweight coding agent that runs in your terminal · GitHub"
[5]: https://developers.openai.com/codex/guides/agents-md "Custom instructions with AGENTS.md – Codex | OpenAI Developers"
[6]: https://code.claude.com/docs/en/agent-sdk/skills "Agent Skills in the SDK - Claude Code Docs"
[7]: https://github.com/alirezarezvani/claude-skills "GitHub - alirezarezvani/claude-skills: 232+ Claude Code skills & agent plugins for Claude Code, Codex, Gemini CLI, Cursor, and 8 more coding agents — engineering, marketing, product, compliance, C-level advisory. · GitHub"
[8]: https://github.com/shanraisshan/claude-code-best-practice "GitHub - shanraisshan/claude-code-best-practice: from vibe coding to agentic engineering - practice makes claude perfect · GitHub"
[9]: https://github.com/ai-boost/awesome-harness-engineering "GitHub - ai-boost/awesome-harness-engineering: Awesome list for AI agent harness engineering: tools, patterns, evals, memory, MCP, permissions, observability, and orchestration. · GitHub"
[10]: https://github.com/pygod-team/pygod "GitHub - pygod-team/pygod: A Python Library for Graph Outlier Detection (Anomaly Detection) · GitHub"
[11]: https://github.com/squareRoot3/GADBench "GitHub - squareRoot3/GADBench: \"GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection\" in NeurIPS 2023 · GitHub"
[12]: https://github.com/squareRoot3/Rethinking-Anomaly-Detection "GitHub - squareRoot3/Rethinking-Anomaly-Detection: \"Rethinking Graph Neural Networks for Anomaly Detection\" in ICML 2022 · GitHub"
[13]: https://github.com/YingtongDou/CARE-GNN "GitHub - YingtongDou/CARE-GNN: Code for CIKM 2020 paper Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters · GitHub"
[14]: https://github.com/safe-graph/graph-fraud-detection-papers "GitHub - safe-graph/graph-fraud-detection-papers: A curated list of Graph/Transformer-based fraud, anomaly, and outlier detection papers & resources · GitHub"

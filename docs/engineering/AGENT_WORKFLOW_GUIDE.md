下面给你一份 **GReaD-Core 实操落地手册**。它不是“提示词合集”，而是一套可以真正执行的 **AI coding agent 工程流程**：如何安装 Claude Code / Codex、如何沉淀上下文为 Markdown、如何装 Superpowers / Ralph、如何配置 skills / subagents / agent teams、如何保持上下文干净、如何让多个 agent 协作而不写偏、以及如何一步步贯彻实验设计。

我会默认你的主力工具是 **Claude Code，简称 cc**；Codex 作为第二实现/审查工具；GitHub + pytest + CI 作为最终裁判。

---

# 0. 先明确总体策略

GReaD-Core 这种研究代码不能靠“让 Agent 一口气写完”。正确流程是：

```text
研究方案 → Markdown 契约 → AGENTS/CLAUDE 指令 → Skills → Subagents → CI/Hooks → Ralph 长循环
```

也就是：

```text
人类负责研究边界和验收标准；
Agent 负责在边界内高速实现；
测试、CI、paper-alignment checks 负责裁判。
```

上一轮整合出来的研究痛点是非常明确的：**Evidence Contract Verifier、score-blind MEP、evidence-conditioned residual reasoner、LLM-free inference、tri-CEC / non-redundancy evaluation**。这些必须变成代码仓库中的长期约束，而不是每次开新会话都重新口头解释。你上传的多 AI 评审意见里也反复强调 verifier、score leakage、CEC、训练流程、跨 detector 适配这些攻击面，因此整个工程 harness 要围绕这些点锁死。

---

# 1. 工具安装与推荐组合

## 1.1 安装 Claude Code

macOS / Linux / WSL 推荐：

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Windows PowerShell：

```powershell
irm https://claude.ai/install.ps1 | iex
```

验证：

```bash
claude --version
claude doctor
```

Claude Code 官方安装文档说明它支持 macOS、Windows、Ubuntu、Debian、Alpine 等环境，并建议安装后通过 `claude --version` 和 `claude doctor` 检查安装状态。([Claude][1])

你可以给它设置一个短别名：

```bash
echo "alias cc='claude'" >> ~/.zshrc
source ~/.zshrc
```

之后我下面写 `cc` 的地方，你也可以替换成 `claude`。

---

## 1.2 安装 Codex CLI 作为第二 Agent

Codex 适合做：

```text
第二实现意见
代码审查
测试补充
AGENTS.md 兼容检查
与 Claude Code 交叉验证
```

安装：

```bash
npm i -g @openai/codex
codex
```

升级：

```bash
npm i -g @openai/codex@latest
```

OpenAI 官方 Codex CLI 文档说明，Codex 是可在本地终端运行的 coding agent，能读取、修改并运行所选目录中的代码，首次运行会要求使用 ChatGPT 账号或 API key 登录。([OpenAI 开发者][2])

---

# 2. 先把研究方案固化为 Markdown 上下文

答案是：**必须转为 Markdown，而且要分层，不要把所有内容塞进一个巨大的 CLAUDE.md。**

推荐结构：

```text
gread-core/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── docs/
│   ├── research/
│   │   ├── GREAD_CORE_FINAL_SCHEME.md
│   │   ├── REVIEWER_PAIN_POINTS.md
│   │   └── PAPER_CLAIMS_AND_NON_CLAIMS.md
│   ├── engineering/
│   │   ├── IMPLEMENTATION_BLUEPRINT.md
│   │   ├── AGENT_WORKFLOW_GUIDE.md
│   │   ├── CONTEXT_HYGIENE.md
│   │   └── EXPERIMENT_LIFECYCLE.md
│   └── decisions/
│       ├── ADR_0001_score_blind_mep.md
│       ├── ADR_0002_evidence_contract_verifier.md
│       ├── ADR_0003_llm_free_inference.md
│       └── ADR_0004_experimental_features.md
├── specs/
│   ├── 001_score_blind_mep.md
│   ├── 002_detector_adapter_protocol.md
│   ├── 003_evidence_contract_verifier.md
│   ├── 004_llm_teacher_and_err.md
│   ├── 005_student_reasoner.md
│   ├── 006_training_protocol.md
│   ├── 007_evaluation_protocol.md
│   └── 008_ablation_matrix.md
├── .claude/
│   ├── settings.json
│   ├── skills/
│   ├── agents/
│   ├── rules/
│   └── hooks/
└── .agents/
    └── skills/
```

## 每个文件放什么

`docs/research/GREAD_CORE_FINAL_SCHEME.md`
放最终研究方案，包含：score-blind MEP、ECV、adapter protocol、reasoner、loss、evaluation。

`docs/research/REVIEWER_PAIN_POINTS.md`
放 18 条改进意见归并结果，尤其是：

```text
verifier 太弱
prediction_score 泄漏
CEC 断裂
训练冷启动
跨 detector 可比性
```

`docs/research/PAPER_CLAIMS_AND_NON_CLAIMS.md`
专门写“主张”和“禁止主张”：

```text
可以说：
- contract-consistent
- score-blind
- LLM-free inference
- detector-adaptable when native evidence exists
- counterfactually responsive

不能说：
- causally guaranteed
- universal any-detector
- LLM rationale is ground truth
- verifier proves semantic truth
```

`docs/engineering/IMPLEMENTATION_BLUEPRINT.md`
放上一轮代码实现方案。

`docs/engineering/AGENT_WORKFLOW_GUIDE.md`
放本文这套实操指南。

`specs/*.md`
每个模块一份 specification，让 Agent 修改代码时只读相关 spec。

`AGENTS.md`
给 Codex、通用 coding agent、GitHub Agent 用。

`CLAUDE.md`
给 Claude Code 用，尽量短，只 import `AGENTS.md` 并补 Claude 特定规则。

Claude Code 官方记忆文档明确建议：`CLAUDE.md` 适合放项目架构、构建命令、编码标准等每次会话都要知道的事实；如果条目变成多步骤过程或只对某部分代码重要，就应移到 skill 或路径范围规则中。文档还建议每个 `CLAUDE.md` 目标保持在 200 行以下，因为它会进入每个会话上下文。([Claude][3])

---

# 3. 配置 AGENTS.md 与 CLAUDE.md

## 3.1 根目录 AGENTS.md

这个文件是所有 agent 的“宪法”。

````markdown
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

## 3.2 CLAUDE.md

Claude Code 读取的是 `CLAUDE.md`，不是 `AGENTS.md`。官方文档建议，如果仓库已经有 `AGENTS.md`，可以在 `CLAUDE.md` 中导入它，从而让两个工具共享同一套约束。:contentReference[oaicite:5]{index=5}

根目录创建：

```markdown
# CLAUDE.md

@AGENTS.md

## Claude Code Specific Rules

- Use Plan Mode before any task that touches more than one module.
- Prefer project skills under .claude/skills/ when available.
- Use subagents for exploration, review, and log-heavy work.
- Keep the main conversation clean: do not paste long logs; write logs to artifacts/logs and summarize.
- After every implementation task, update docs/engineering/PROJECT_STATE.md.
- Never claim completion until required validation commands pass or the failure is documented.
````

---

# 4. 安装 Superpowers、Ralph、项目 Skill

## 4.1 安装 Superpowers

我建议安装 **Superpowers**，因为它的价值不是“让模型更聪明”，而是把 brainstorm、TDD、debug、review、execute-plan 这些工程纪律变成可复用 workflow。Anthropic 官方插件页也说明，Superpowers 提供 TDD、系统化调试、头脑风暴、subagent-driven development、内置代码审查等能力。([Claude][4])

在 Claude Code 中执行：

```text
/plugin install superpowers@claude-plugins-official
```

也可以用 Superpowers 自己的 marketplace：

```text
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
```

Superpowers GitHub README 也列出官方 Claude 插件市场安装方式，以及适配 Codex、Cursor、OpenCode、Gemini CLI 等平台的方式。([GitHub][5])

安装后常用命令：

```text
/brainstorming
/execute-plan
/debug
```

我的建议：

```text
需求探索：/brainstorming
实现大任务：/execute-plan
定位失败测试：/debug
长期自动迭代：Ralph Loop，而不是单纯 /execute-plan 无限跑
```

---

## 4.2 安装 Ralph Loop

Ralph 适合“有明确完成标准、可以反复跑测试、失败后继续修”的长任务。Anthropic 官方 Ralph Loop 插件页说明，它通过 stop hook 自动重新投喂 prompt，并保留文件修改与 git history，使 Claude 可以基于测试失败和前次尝试持续改进。官方示例命令是：

```text
/ralph-loop "your prompt here" --max-iterations 10 --completion-promise "DONE"
```

([Claude][6])

安装：

```text
/plugin install ralph-loop@claude-plugins-official
```

推荐使用方式：

```text
/ralph-loop "Implement Epic 1: schemas and verifier only.
Completion criteria:
1. tests/unit/test_mep_score_blind.py passes
2. tests/unit/test_err_schema.py passes
3. tests/unit/test_contract_verifier.py passes
4. tests/paper_alignment/test_prediction_score_not_in_prompt.py passes
5. ruff check . passes
Do not implement detector adapters, LLM teacher, or models in this loop.
When complete, print GREAD_EPIC1_DONE." --max-iterations 8 --completion-promise "GREAD_EPIC1_DONE"
```

关键是：**Ralph 只适合已经被切成清晰 Epic 的任务，不适合拿来“帮我实现整篇论文”。**

---

## 4.3 Super-Ralph 要不要装？

有一个社区项目 **Super-Ralph**，把 Ralph 的长循环和 Superpowers 的方法论合在一起。它的 README 说 Ralph 提供“耐力”，Superpowers 提供“工程纪律”，两者合并成自主开发循环。([GitHub][7])

但我建议你第一阶段不要用 Super-Ralph，先用官方市场的：

```text
superpowers
ralph-loop
```

原因是：

```text
1. 官方插件更容易维护；
2. 两者分开更容易定位问题；
3. 研究代码最怕自动化工具链本身带来不确定性；
4. 你需要先把 GReaD-Core 的 harness 跑稳。
```

等 Epic 0–3 稳定后，再考虑 Super-Ralph。

---

## 4.4 Ultrawork 开不开？

**默认不开。**

Ultrawork 的一个公开 GitHub 仓库已经标注为 deprecated，并说明 Claude Code 内置 Team tools 已提供原生多 agent orchestration，因此该 skill 不再维护。([GitHub][8])

你的项目不是普通 CRUD，而是科研代码，要求严格对齐论文方案。Ultrawork 这种“自动路由 + 高自治”的工具会提高跑偏风险。建议：

```text
不用于主线实现；
不用于论文核心模块；
最多用于临时分支上的探索性 refactor；
一旦使用，必须先 git branch / worktree 隔离。
```

替代方案：

```text
主线工程纪律：Superpowers
长循环：Ralph Loop
并行协作：Claude Code Agent Teams 或自定义 subagents
上下文隔离：subagents + git worktree
```

---

# 5. 安装项目专用 Skill

## 5.1 Claude Code 项目 Skill

创建：

````bash
mkdir -p .claude/skills/gread-core-implementer
cat > .claude/skills/gread-core-implementer/SKILL.md <<'EOF'
---
description: Implement and verify GReaD-Core research code while preserving strict paper alignment.
---

# GReaD-Core Implementer Skill

Use this skill when implementing, refactoring, testing, or reviewing code in this repository.

## Research Contract

GReaD-Core consists of:

1. Score-blind Minimal Evidence Package
2. Detector-Evidence Adapter Protocol
3. Evidence Rationale Record
4. Evidence Contract Verifier
5. Offline LLM teacher generation
6. Evidence-conditioned residual student reasoner
7. Stage 1 / Stage 2 / Stage 3 training protocol
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
- Experimental only

## Non-Negotiable Rules

- Never expose prediction_score to the LLM teacher.
- Never use ERR summary as training signal.
- Never allow rejected ERR into reasoning loss.
- Never import LLM code in inference.
- Never modify risk taxonomy without updating schemas, contracts, tests, README, and specs.
- Never enable DHEF, CER, ECB, adaptive lambda, or multi-sample self-consistency in main configs.

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
  EOF

````

Claude Code skills 是一个含 `SKILL.md` 的目录，`description` 用于自动触发，项目级 skills 放在 `.claude/skills/<skill-name>/SKILL.md`，个人级 skills 放在 `~/.claude/skills/`；Claude Code 会按相关性自动使用，也可以通过 slash command 直接调用。:contentReference[oaicite:11]{index=11}

你使用时可以显式写：

```text
Use the gread-core-implementer skill.
Implement only the Evidence Contract Verifier module.
````

---

## 5.2 Codex 项目 Skill

Codex 的 skills 推荐放在 `.agents/skills`，skill 必须有 `name` 和 `description`。Codex 会先看到 skill 的 name、description、路径，只有决定使用时才加载完整 `SKILL.md`，这是它的 progressive disclosure 机制。([OpenAI 开发者][9])

创建：

```bash
mkdir -p .agents/skills/gread-core-implementer
cp .claude/skills/gread-core-implementer/SKILL.md .agents/skills/gread-core-implementer/SKILL.md
```

然后可以在 Codex 里说：

```text
Use $gread-core-implementer to review whether this PR violates score-blind MEP or LLM-free inference constraints.
```

---

# 6. 创建 Subagents

Subagents 的作用不是“越多越好”，而是 **隔离上下文污染**。Claude 官方文档说，subagent 在自己的 context window 中运行，适合把搜索结果、日志、文件内容等高容量操作留在主对话之外，只把摘要返回。([Claude][10])

## 6.1 推荐 6 个项目级 subagents

放在：

```text
.claude/agents/
```

### 1. research-alignment-reviewer

```bash
mkdir -p .claude/agents
cat > .claude/agents/research-alignment-reviewer.md <<'EOF'
---
name: research-alignment-reviewer
description: Reviews whether code changes remain aligned with the GReaD-Core paper design and non-claims.
tools: Read, Grep, Glob, Bash
model: opus
---

You are the research alignment reviewer for GReaD-Core.

Focus on:
- score-blind MEP
- Evidence Contract Verifier
- LLM-free inference
- accepted ERR only
- summary not used for training
- main method vs experimental boundary
- claims vs non-claims

Reject any change that:
- leaks prediction_score into prompts or evidence targets
- imports LLM/network code in inference
- treats ERR summary as a label
- changes the training objective without marking it experimental
- silently modifies risk taxonomy
EOF
```

### 2. verifier-engineer

```bash
cat > .claude/agents/verifier-engineer.md <<'EOF'
---
name: verifier-engineer
description: Implements and tests deterministic Evidence Contract Verifier components.
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
---

You implement only verifier-related code:
- schema validity
- evidence availability
- role consistency
- risk-evidence contracts
- score-blindness
- label compatibility

Do not implement models, adapters, training, or LLM teacher code.

Always add or update tests under:
- tests/unit/test_contract_verifier.py
- tests/paper_alignment/
EOF
```

### 3. pytorch-architect

```bash
cat > .claude/agents/pytorch-architect.md <<'EOF'
---
name: pytorch-architect
description: Designs and implements PyTorch modules for evidence-conditioned GReaD-Core reasoner.
tools: Read, Grep, Glob, Edit, Write, Bash
model: sonnet
---

You implement PyTorch model code only:
- EvidenceEncoder
- GReaDReasoner
- signed evidence heads
- evidence-gated residual readout

Constraints:
- no LLM imports
- no online network imports
- all tensor shapes documented
- rho=0 must recover base detector logits
- tests must include shape and gradient sanity checks
EOF
```

### 4. experiment-runner

```bash
cat > .claude/agents/experiment-runner.md <<'EOF'
---
name: experiment-runner
description: Runs smoke tests, experiment scripts, metric exports, and diagnoses failures without changing core method code.
tools: Read, Grep, Glob, Bash
model: sonnet
---

You run and diagnose experiments.

You may:
- run tests
- inspect logs
- summarize failures
- recommend fixes

You must not:
- edit core method code
- change research assumptions
- silently skip tests
- claim success when metrics are missing
EOF
```

### 5. reproducibility-auditor

```bash
cat > .claude/agents/reproducibility-auditor.md <<'EOF'
---
name: reproducibility-auditor
description: Audits configs, seeds, artifact paths, cache hashes, checkpoints, and README reproduction steps.
tools: Read, Grep, Glob, Bash, Edit, Write
model: sonnet
---

Audit reproducibility:
- seed handling
- config hashing
- dataset split hashing
- ERR cache hashing
- checkpoint metadata
- README commands
- artifact directory layout

Do not change model logic.
EOF
```

### 6. code-reviewer

如果你装了 Superpowers，它通常自带 code review 能力；但项目级 reviewer 仍有价值：

```bash
cat > .claude/agents/code-reviewer.md <<'EOF'
---
name: code-reviewer
description: Reviews modified code for maintainability, typing, tests, and paper-alignment risks.
tools: Read, Grep, Glob, Bash
model: sonnet
---

Review changed files only.

Check:
- type hints
- test coverage
- config-driven constants
- no hidden dependencies
- no score leakage
- no LLM in inference
- no summary-as-label
- no untested verifier rules
EOF
```

你也可以用 `/agents` 交互式创建这些 subagents；Claude Code 官方推荐通过 `/agents` 管理，项目级 subagents 放 `.claude/agents/`，用户级放 `~/.claude/agents/`，项目级适合签入版本控制供团队共享。([Claude][10])

---

# 7. 多 Agent 协作方式怎么选

## 7.1 三种协作方式

| 方式               | 用途                                 | 适合 GReaD-Core 哪些任务            |
| ---------------- | ---------------------------------- | ----------------------------- |
| Subagents        | 主会话内派出独立上下文 worker，结果回到主会话         | 搜索、审查、日志分析、单模块实现              |
| Agent Teams      | 多个 Claude Code 实例组成团队，有共享任务列表和互相通信 | 大型并行审查、跨模块设计、多个实验方案对抗         |
| Git worktree 多会话 | 手动多开多个隔离目录                         | 并行开发 adapter、eval、docs，避免文件冲突 |

Agent Teams 是实验功能，默认关闭；官方文档说明它适合“并行探索有价值”的任务，例如研究与审查、新模块、debugging competing hypotheses、跨层协作，但 token 成本更高，并且存在恢复、任务协调、关闭行为等限制。([Claude][11])

---

## 7.2 推荐策略

### 平时默认

```text
单 Claude Code 主会话 + custom subagents
```

适合：

```text
Epic 1 verifier
Epic 2 MEP
Epic 3 adapter
Epic 4 LLM teacher
Epic 5 reasoner
```

### 需要并行审查时

```text
Agent Team
```

适合：

```text
PR 合并前
实验设计审查
跨 detector adapter protocol 审查
tri-CEC / non-redundancy correctness 审查
```

### 需要同时实现多个互不冲突模块时

```text
git worktree + 多 Claude Code 会话
```

例如：

```text
worktree A: verifier
worktree B: evaluation metrics
worktree C: docs / reproducibility
```

---

## 7.3 开启 Agent Teams

`.claude/settings.json`：

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "teammateMode": "in-process"
}
```

官方文档说明，启用 Agent Teams 需要设置 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`，并且每个 teammate 是独立 Claude Code session，有自己的上下文窗口；teams 适合 3–5 个 teammate，且要避免多个 teammate 编辑同一文件。([Claude][11])

### 推荐 Agent Team prompt

```text
Create an agent team for a research-code review of GReaD-Core Epic 1.

Spawn four teammates:
1. verifier-soundness reviewer: focus on ECV rules and label compatibility
2. leakage reviewer: search for prediction_score leakage and LLM in inference
3. testing reviewer: inspect pytest coverage and missing fixtures
4. maintainability reviewer: inspect typing, configs, and module boundaries

Require plan approval before any teammate suggests edits.
Do not allow teammates to edit files.
They should produce findings only.
The lead must synthesize a final actionable checklist.
```

---

# 8. 上下文如何保持干净

这是长期项目能不能跑稳的关键。

## 8.1 主对话只保留 5 类内容

主会话只应该出现：

```text
1. 当前 Epic 的目标
2. 当前验收标准
3. Agent 返回的短摘要
4. 测试结果摘要
5. 你的人类决策
```

不要在主对话粘贴：

```text
长日志
完整 traceback
大段代码
完整论文方案
完整 experiment output
LLM ERR cache
```

这些应该写入文件：

```text
artifacts/logs/
artifacts/metrics/
docs/engineering/PROJECT_STATE.md
docs/decisions/
```

## 8.2 让 subagent 承担污染性上下文

比如：

```text
Use experiment-runner to run pytest tests/unit/test_contract_verifier.py.
Return only:
- command run
- pass/fail
- top 3 failure causes
- files likely responsible
Do not paste full logs into the main conversation.
```

这符合 Claude 官方对 subagents 的定位：当辅助任务会用搜索结果、日志或文件内容污染主对话时，让 subagent 在独立上下文中完成并返回摘要。([Claude][10])

## 8.3 每个 Epic 一个新会话

推荐节奏：

```text
Session 0: bootstrap repo and docs
Session 1: schemas + verifier
Session 2: MEP + leakage guard
Session 3: adapters
Session 4: LLM teacher cache
Session 5: reasoner + losses
Session 6: training smoke
Session 7: evaluation metrics
Session 8: ablations + tables
```

每个 session 结束前让 Agent 更新：

```text
docs/engineering/PROJECT_STATE.md
docs/engineering/NEXT_TASKS.md
docs/decisions/ADR_xxxx_*.md
```

## 8.4 使用 `/compact`，但不要依赖它救场

当会话很长时可以用 `/compact` 压缩上下文。但更好的做法是：

```text
先把状态写入 PROJECT_STATE.md；
再开新会话；
让新会话读取 CLAUDE.md + PROJECT_STATE.md + 当前 spec。
```

---

# 9. 记忆如何持久化

## 9.1 四层记忆

| 层         | 文件 / 机制                             | 作用           |
| --------- | ----------------------------------- | ------------ |
| 项目硬约束     | `AGENTS.md`, `CLAUDE.md`            | 每次会话加载的规则    |
| 模块规格      | `specs/*.md`                        | 按需读取的研究/实现契约 |
| 动态项目状态    | `docs/engineering/PROJECT_STATE.md` | 当前完成度、失败、下一步 |
| Agent 自学习 | Claude 自动记忆、subagent memory         | 个人/工作树习惯与发现  |

Claude Code 官方文档说，每个会话都是新上下文，跨会话主要靠两种机制：你写的 `CLAUDE.md` 和 Claude 自动维护的记忆；subagents 也可以维护自己的自动记忆。([Claude][3])

## 9.2 PROJECT_STATE.md 模板

````markdown
# PROJECT_STATE.md

## Current Phase

Epic: 1 - Schemas and Evidence Contract Verifier
Branch: epic1-verifier
Last updated: 2026-04-30

## Completed

- Created Pydantic schemas for MEP and ERR.
- Implemented to_teacher_payload() excluding calibration channel.
- Added initial verifier config.

## Current Failures

- test_contract_verifier.py::test_label_compatibility_benign_rejects_strong_risk failing
- Cause: label compatibility rule not applied when label is None vs int

## Next Tasks

1. Fix label compatibility handling.
2. Add fixture for detector_signal=unavailable spectral anomaly rejection.
3. Run paper alignment tests.

## Research Constraints Reconfirmed

- prediction_score is calibration-only.
- summary not used for training.
- rejected ERR excluded from reasoning loss.
- no LLM imports in inference.

## Commands Last Run

```bash
ruff check .
pytest tests/unit/test_contract_verifier.py
````

````

## 9.3 每次关闭会话前的 prompt

```text
Before we stop, update docs/engineering/PROJECT_STATE.md.

Include:
1. What changed
2. What passed
3. What failed
4. What files were touched
5. Which research constraints were checked
6. Exact next prompt I should use in the next session

Do not start new implementation work.
````

---

# 10. 权限模式、模型模式怎么选

## 10.1 Plan Mode

复杂任务先开 Plan Mode。

启动：

```bash
cc --model opusplan --permission-mode plan
```

或在会话中按 `Shift+Tab` 切换：

```text
Normal Mode → Auto-Accept Mode → Plan Mode
```

Plan Mode 是只读模式，适合多文件变更前先研究代码库、制定计划；官方文档也建议对多步实现、代码探索、安全审查先用 Plan Mode，然后满意后切换到 Normal Mode 执行。([claudecn.com][12])

推荐第一条 prompt：

```text
ultrathink.

We are implementing GReaD-Core.
Do not edit files yet.

Read:
- AGENTS.md
- docs/research/GREAD_CORE_FINAL_SCHEME.md
- docs/engineering/IMPLEMENTATION_BLUEPRINT.md
- specs/003_evidence_contract_verifier.md

Create a detailed implementation plan for Epic 1 only:
schemas + risk taxonomy + Evidence Contract Verifier.

The plan must include:
1. target files
2. tests to write first
3. exact acceptance criteria
4. commands to run
5. forbidden shortcuts
6. where paper-alignment risks may occur

Do not implement until I approve.
```

## 10.2 模型选择

推荐：

| 场景                 | 模型                              |
| ------------------ | ------------------------------- |
| 大架构、研究方案、Plan Mode | `opusplan` 或 `opus`             |
| 日常编码、测试修复          | `sonnet`                        |
| 快速搜索、只读探索          | `haiku` subagent                |
| 长会话需要大上下文          | `opus[1m]` 或 `sonnet[1m]`，看账号支持 |

Claude Code 模型配置文档说明：`sonnet` 用于日常编码任务，`opus` 用于复杂推理，`haiku` 用于快速简单任务，`opusplan` 会在 Plan Mode 用 Opus、执行时切换到 Sonnet。文档还说明提示中包含 `ultrathink` 会在该轮提示模型进行更多推理，但不会改变 API 级别的 effort 设置。([Claude][13])

## 10.3 权限模式

`.claude/settings.json` 推荐：

```json
{
  "model": "opusplan",
  "permissions": {
    "defaultMode": "plan",
    "allow": [
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(ruff check *)",
      "Bash(mypy src)",
      "Bash(pytest *)",
      "Bash(python scripts/check_no_leakage.py)",
      "Bash(python scripts/check_no_llm_inference.py)",
      "Bash(bash scripts/run_smoke.sh)"
    ],
    "deny": [
      "Bash(git push *)",
      "Bash(rm -rf *)",
      "Read(.env)",
      "Read(**/.env)",
      "Read(**/*secret*)"
    ]
  },
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  },
  "teammateMode": "in-process"
}
```

Claude Code 权限文档说明，权限规则按 `deny -> ask -> allow` 顺序评估，deny 始终优先；`plan` 模式只允许分析不能修改，`acceptEdits` 会自动接受工作目录内文件编辑，`auto` 仍是研究预览，`bypassPermissions` 只应在容器或虚拟机等隔离环境中使用。([Claude][14])

我的建议：

```text
Plan Mode：所有 Epic 开始前必用
Normal Mode：主要实现模式
acceptEdits：仅在任务很小、测试完备时用
auto：暂不建议用于主线科研代码
bypassPermissions：只在 disposable container / throwaway worktree 中用
```

---

# 11. Hooks：用自动门禁防止 Agent 跑偏

Claude Code hooks 可以在生命周期事件触发时执行命令或提示，`TaskCompleted` hook 可以在任务标记完成时强制测试/lint 等完成标准，不通过时阻止任务关闭。([Claude][15])

## 11.1 创建 hook 脚本

```bash
mkdir -p .claude/hooks
cat > .claude/hooks/paper-alignment-gate.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

echo "[paper-alignment-gate] running..."

ruff check .
mypy src
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py

echo "[paper-alignment-gate] passed"
EOF

chmod +x .claude/hooks/paper-alignment-gate.sh
```

## 11.2 配置 TaskCompleted hook

`.claude/settings.json` 添加：

```json
{
  "hooks": {
    "TaskCompleted": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/paper-alignment-gate.sh",
            "timeout": 300
          }
        ]
      }
    ]
  }
}
```

如果这个 hook 退出码为 2，Claude Code 会把 stderr 作为反馈，并阻止任务被标记完成；这适合 “完成任务前必须通过 paper alignment gate” 的场景。([Claude][15])

---

# 12. 最推荐的落地路线

## Phase 0：仓库与上下文启动

目标：

```text
先让 Agent 被约束住，再让它写代码。
```

命令：

```bash
mkdir gread-core
cd gread-core
git init
mkdir -p docs/research docs/engineering docs/decisions specs src/gread_core tests scripts configs artifacts
```

打开 Claude Code：

```bash
cc --model opusplan --permission-mode plan
```

输入：

```text
ultrathink.

We are bootstrapping the GReaD-Core research codebase.

Do not implement model logic yet.

Create the repository harness only:
1. AGENTS.md
2. CLAUDE.md
3. docs/research/GREAD_CORE_FINAL_SCHEME.md
4. docs/engineering/IMPLEMENTATION_BLUEPRINT.md
5. docs/engineering/PROJECT_STATE.md
6. specs/*.md placeholders
7. pyproject.toml with ruff, mypy, pytest
8. tests/paper_alignment placeholders
9. scripts/check_no_leakage.py
10. scripts/check_no_llm_inference.py

Research constraints:
- prediction_score calibration-only
- LLM offline only
- inference LLM-free
- verifier deterministic
- summary not used for training
- DHEF/CER/ECB experimental only

First produce a file-by-file plan.
Do not edit files until I approve.
```

批准计划后：

```text
Proceed with the bootstrap exactly as planned.
After editing, run:
ruff check .
pytest tests/paper_alignment
```

---

## Phase 1：Schemas + Verifier

启动：

```bash
git checkout -b epic1-schemas-verifier
cc --model opusplan --permission-mode plan
```

Prompt：

```text
Use the gread-core-implementer skill.

Implement Epic 1 only:
- Pydantic schemas for MEP and ERR
- risk taxonomy
- Evidence Contract Verifier
- verifier config YAML
- unit tests
- paper-alignment tests

Read:
- AGENTS.md
- specs/001_score_blind_mep.md
- specs/003_evidence_contract_verifier.md
- docs/research/PAPER_CLAIMS_AND_NON_CLAIMS.md

Forbidden:
- do not implement detector adapters
- do not implement LLM teacher
- do not implement PyTorch models
- do not add experimental DHEF/CER/ECB

First write tests, then implement.

Acceptance:
1. prediction_score excluded from teacher payload
2. counter_signal rejected as supporting evidence
3. prediction_score rejected if cited as evidence
4. detector_signal=unavailable rejects spectral_anomaly
5. label compatibility works
6. summary not in training_targets
7. all paper alignment tests pass
```

适合用 Ralph：

```text
/ralph-loop "Implement Epic 1 schemas and verifier only.
Completion criteria:
- ruff check . passes
- mypy src passes
- pytest tests/unit/test_mep_score_blind.py passes
- pytest tests/unit/test_err_schema.py passes
- pytest tests/unit/test_contract_verifier.py passes
- pytest tests/paper_alignment passes
- python scripts/check_no_leakage.py passes
- python scripts/check_no_llm_inference.py passes

Forbidden:
- no detector adapters
- no LLM teacher
- no PyTorch model implementation
- no experimental DHEF/CER/ECB

When complete, print GREAD_EPIC1_DONE." --max-iterations 8 --completion-promise "GREAD_EPIC1_DONE"
```

---

## Phase 2：MEP + Adapter Protocol

不要直接实现所有 detector。先做：

```text
Base EvidenceAdapter
Generic evidence signals
Mock adapter
BWGNN adapter stub
Adapter protocol tests
```

Prompt：

```text
Use the gread-core-implementer skill.

Implement Epic 2 only: score-blind MEP builder and detector adapter protocol.

Read:
- specs/001_score_blind_mep.md
- specs/002_detector_adapter_protocol.md
- tests from Epic 1

Target behavior:
- every adapter outputs MinimalEvidencePackage
- MEP has calibration and reasoning channels
- to_teacher_payload() excludes calibration
- adapter outputs generic + detector_native + counter evidence
- no prediction_score in allowed_support_ids

Implement:
- src/gread_core/adapters/base.py
- src/gread_core/evidence/mep.py
- src/gread_core/evidence/generic_signals.py
- tests/integration/test_adapter_protocol.py

Do not implement LLM teacher or training.
```

---

## Phase 3：Offline LLM teacher + cache

这一步最容易把 score leak 进去，必须严控。

Prompt：

```text
Use the gread-core-implementer skill.

Implement Epic 3 only: offline LLM teacher and ERR cache.

Read:
- specs/004_llm_teacher_and_err.md
- AGENTS.md
- tests/paper_alignment/test_prediction_score_not_in_prompt.py

Requirements:
1. PromptBuilder may only use mep.to_teacher_payload()
2. prompt template must not contain prediction_score, fraud score, probability, or base score
3. LLM client abstraction must be isolated under src/gread_core/llm
4. cache must be keyed by prompt hash
5. replay mode must run without network
6. generated ERR must pass Evidence Contract Verifier before becoming a training target

Do not import gread_core.llm anywhere under src/gread_core/inference.
```

---

## Phase 4：Reasoner + Loss

Prompt：

```text
Use the gread-core-implementer skill.

Implement Epic 4 only:
- EvidenceEncoder
- GReaDReasoner
- signed evidence masks
- evidence-gated residual readout
- reasoning distillation loss

Read:
- specs/005_student_reasoner.md
- specs/006_training_protocol.md

Acceptance:
1. forward returns base_logit, final_logit, type_logits, pos_mask_logits, neg_mask_logits
2. rho=0 recovers base_logit as final_logit
3. rejected ERR samples have zero type/evidence loss
4. summary is not used
5. no LLM or network imports in models or inference
6. shape tests pass
```

---

## Phase 5：Training pipeline

Prompt：

```text
Use the gread-core-implementer skill.

Implement Epic 5 only:
- Stage 1 train base detector
- Stage 2 generate and verify ERR
- Stage 3 train reasoner
- checkpoint metadata
- tiny graph smoke test

Read:
- specs/006_training_protocol.md
- docs/engineering/EXPERIMENT_LIFECYCLE.md

Acceptance:
1. Stage 1 does not call LLM
2. Stage 2 is the only stage allowed to call LLM
3. Stage 3 trains with accepted ERR only
4. smoke training runs on CPU
5. artifact metadata includes seed, config hash, split hash, err cache hash
```

---

## Phase 6：Evaluation

Prompt：

```text
Use the gread-core-implementer skill.

Implement Epic 6 only:
- detection metrics
- reasoning metrics
- tri-CEC
- non-redundancy test
- ablation runner
- table exporter

Read:
- specs/007_evaluation_protocol.md
- specs/008_ablation_matrix.md

Acceptance:
1. ROC-AUC, AUPRC, Recall@K, Precision@K, F1 implemented
2. verifier acceptance rate and contract violation rate reported
3. CEC-score, CEC-type, CEC-evidence implemented
4. non-redundancy reports Y~P, Y~P+T, Y~P+T+M
5. ablation configs do not alter main method defaults
```

---

# 13. Superpowers 怎么用

建议你这样用：

## 13.1 研究方案转规格

```text
/brainstorming

We need to convert the final GReaD-Core research plan into implementation specs.
Do not write code.
Help me refine:
1. module boundaries
2. test-first acceptance criteria
3. paper-alignment checks
4. main method vs experimental boundaries
5. minimal MVP sequence
```

## 13.2 执行一个已批准计划

```text
/execute-plan

Execute only Epic 1 from docs/engineering/IMPLEMENTATION_BLUEPRINT.md.
Follow AGENTS.md strictly.
Use TDD.
Do not expand scope.
Stop after validation commands and update PROJECT_STATE.md.
```

## 13.3 Debug

```text
/debug

pytest tests/unit/test_contract_verifier.py is failing.
Follow systematic debugging.
Do not patch randomly.
First identify root cause, then propose minimal fix, then run the test.
```

Superpowers 官方插件说明里强调，它会引导 red-green-refactor TDD、系统化调试、代码审查、brainstorming 和 subagent-driven development；这正适合 GReaD-Core 的 harness-first 落地方式。([Claude][4])

---

# 14. Ralph 怎么用

## 14.1 Ralph 适合什么

适合：

```text
有明确边界
有自动测试
有完成信号
能失败后继续修
不需要你不断做研究决策
```

不适合：

```text
还没定设计的模块
跨 5 个 Epic 的大任务
需要判断论文贡献边界的任务
需要人工选择实验策略的任务
```

## 14.2 Ralph prompt 模板

```text
/ralph-loop "Task: <one Epic only>.

Read:
- AGENTS.md
- CLAUDE.md
- specs/<current_spec>.md
- docs/engineering/PROJECT_STATE.md

Scope:
- implement only <module>
- do not touch <forbidden modules>

Acceptance commands:
- ruff check .
- mypy src
- pytest <specific tests>
- pytest tests/paper_alignment
- python scripts/check_no_leakage.py
- python scripts/check_no_llm_inference.py

Completion criteria:
- all acceptance commands pass
- docs/engineering/PROJECT_STATE.md updated
- final response contains <COMPLETION_TOKEN>

If any acceptance command cannot pass, document the failure and do not print the completion token." --max-iterations 8 --completion-promise "<COMPLETION_TOKEN>"
```

示例：

```text
/ralph-loop "Task: Implement score-blind MEP and leakage guards.

Read:
- AGENTS.md
- specs/001_score_blind_mep.md
- docs/research/PAPER_CLAIMS_AND_NON_CLAIMS.md

Scope:
- src/gread_core/schemas/evidence.py
- src/gread_core/evidence/mep.py
- tests/unit/test_mep_score_blind.py
- tests/paper_alignment/test_prediction_score_not_in_prompt.py

Forbidden:
- no LLM teacher
- no detector adapters
- no model code
- no training code

Acceptance commands:
- ruff check .
- mypy src
- pytest tests/unit/test_mep_score_blind.py
- pytest tests/paper_alignment/test_prediction_score_not_in_prompt.py
- python scripts/check_no_leakage.py

Completion criteria:
- all commands pass
- PROJECT_STATE.md updated
- final response contains GREAD_MEP_DONE

If any command fails, keep iterating until fixed or document blocker." --max-iterations 6 --completion-promise "GREAD_MEP_DONE"
```

---

# 15. Codex 怎么配合

## 15.1 用 Codex 做第二审查

在同一个 repo：

```bash
codex
```

Prompt：

```text
Use AGENTS.md.

Review the current diff for GReaD-Core paper alignment.

Check:
1. prediction_score leakage
2. LLM imports in inference
3. rejected ERR entering loss
4. summary used as training signal
5. verifier determinism
6. main method vs experimental boundary
7. missing tests

Do not edit files.
Return a prioritized issue list with file paths.
```

## 15.2 用 Codex 安装 skill

Codex 官方说明，可以用 `$skill-installer` 安装 curated 或 experimental skills；仓库本地 skills 放 `.agents/skills`，用户级放 `~/.agents/skills`。([OpenAI 开发者][9])

例如：

```text
$skill-installer gh-address-comments
```

但对于 GReaD-Core，最重要的是你自己的：

```text
.agents/skills/gread-core-implementer/SKILL.md
```

---

# 16. 实验设计如何贯彻到代码

## 16.1 每个实验都必须有 config

不要让 Agent 在代码里写死：

```python
lambda_reason = 0.5
residual_rho = 0.1
```

必须放：

```text
configs/default.yaml
configs/experiments/main_bwgnn_yelp.yaml
configs/experiments/ablation_score_visible.yaml
configs/experiments/ablation_schema_only_verifier.yaml
configs/experiments/ablation_parallel_heads.yaml
```

## 16.2 主方法 config

```yaml
method:
  score_blind: true
  lambda_reason: 0.5
  residual_rho: 0.1
  signed_evidence_masks: true
  use_llm_at_inference: false

experimental:
  dhef: false
  cer_regularizer: false
  evidence_conflict_bucket: false
  adaptive_lambda: false
  multi_sample_self_consistency: false
```

## 16.3 消融 config 必须显式标注

```yaml
paper_warning:
  purpose: "Ablation only. Not part of the main method."
```

## 16.4 实验表输出必须自动化

每次实验写：

```text
artifacts/metrics/<experiment_id>/detection.json
artifacts/metrics/<experiment_id>/reasoning.json
artifacts/metrics/<experiment_id>/cec.json
artifacts/metrics/<experiment_id>/non_redundancy.json
artifacts/tables/main_results.csv
artifacts/tables/ablation_results.csv
```

---

# 17. 什么时候开 Plan、什么时候开 Ralph、什么时候开 Agent Team

## 17.1 决策表

| 场景         | 工具 / 模式                                      |
| ---------- | -------------------------------------------- |
| 第一次设计 Epic | `cc --model opusplan --permission-mode plan` |
| 研究方案边界讨论   | Plan Mode + `ultrathink`                     |
| 单模块实现      | Normal Mode + project skill                  |
| 明确测试驱动长任务  | Ralph Loop                                   |
| 失败测试定位     | Superpowers `/debug`                         |
| 代码审查       | code-reviewer subagent                       |
| 大型并行审查     | Agent Team                                   |
| 多模块并行实现    | git worktree + 多 cc session                  |
| 日常第二意见     | Codex                                        |
| 自动高自治总控    | 暂不使用 Ultrawork                               |

## 17.2 我的推荐默认组合

```text
Plan: Claude Code opusplan
Implement: Claude Code sonnet + gread-core skill
Long loop: Ralph Loop
Discipline: Superpowers
Review: Codex + research-alignment-reviewer
Parallel review: Agent Teams
Parallel implementation: git worktrees
```

---

# 18. Git workflow

每个 Epic 一个分支：

```bash
git checkout -b epic1-schemas-verifier
```

每个 Ralph loop 前先 clean：

```bash
git status
```

每次 Agent 完成后：

```bash
ruff check .
mypy src
pytest tests/unit
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py
git diff
git add .
git commit -m "feat: implement evidence contract verifier"
```

并且要求 Agent 写：

```text
docs/engineering/PROJECT_STATE.md
```

---

# 19. 防跑偏的 10 条硬规则

把这 10 条贴在 `AGENTS.md`、`CLAUDE.md`、Skill、PR template 里：

```text
1. 不允许 prediction_score 进入 LLM prompt。
2. 不允许 prediction_score 成为 evidence target。
3. 不允许 inference import gread_core.llm。
4. 不允许 ERR summary 进入 training loss。
5. 不允许 rejected ERR 进入 type/evidence loss。
6. 不允许用 LLM-as-judge 替代 Evidence Contract Verifier。
7. 不允许默认启用 DHEF / CER / ECB / adaptive lambda。
8. 不允许声称 causal guarantee。
9. 不允许没有 config 的超参数。
10. 不允许没有 paper-alignment test 的核心模块 PR。
```

---

# 20. 开发顺序最终版

你真正执行时，按这个顺序：

```text
Day 1:
  Install Claude Code / Codex
  Install Superpowers / Ralph Loop
  Create repo
  Create AGENTS.md / CLAUDE.md / Skill / Subagents
  Create specs and docs
  Add paper-alignment test placeholders

Day 2:
  Epic 1: MEP + ERR schema + ECV
  No PyTorch yet
  No LLM yet

Day 3:
  Epic 2: Adapter protocol + mock/BWGNN adapter stub
  Add leakage guards

Day 4:
  Epic 3: Offline LLM teacher + cache replay
  Ensure prompt is score-blind

Day 5:
  Epic 4: EvidenceEncoder + Reasoner + losses
  Ensure LLM-free inference

Day 6:
  Epic 5: Stage 1/2/3 training smoke
  Tiny graph CPU test

Day 7:
  Epic 6: detection/reasoning/tri-CEC/non-redundancy metrics

After MVP:
  Add BWGNN full
  Add CARE-GNN adapter
  Add tree-neighbor baseline
  Run ablations
  Export paper tables
```

---

# 21. 你接下来第一条该输入的 prompt

打开项目根目录：

```bash
cd gread-core
cc --model opusplan --permission-mode plan
```

输入：

```text
ultrathink.

We are implementing GReaD-Core, a contract-verified score-blind evidence distillation framework for LLM-free graph fraud reasoning.

Do not write code yet.

First, read or create these project context files:
- AGENTS.md
- CLAUDE.md
- docs/research/GREAD_CORE_FINAL_SCHEME.md
- docs/research/REVIEWER_PAIN_POINTS.md
- docs/research/PAPER_CLAIMS_AND_NON_CLAIMS.md
- docs/engineering/IMPLEMENTATION_BLUEPRINT.md
- docs/engineering/PROJECT_STATE.md
- specs/001_score_blind_mep.md
- specs/003_evidence_contract_verifier.md

Your task:
Create a complete bootstrap plan for the repository harness.

The plan must include:
1. exact directory tree
2. files to create
3. what content each file should contain
4. first tests to write
5. CI and hook gates
6. project skills and subagents
7. how to validate score-blind MEP and LLM-free inference
8. what must remain experimental-only

Forbidden:
- no PyTorch model implementation yet
- no LLM client implementation yet
- no detector adapter implementation yet
- no experiments yet

After the plan, ask me to approve before editing.
```

批准后：

```text
Proceed with the bootstrap plan exactly.
Keep changes minimal.
After editing, run:
ruff check .
pytest tests/paper_alignment
python scripts/check_no_leakage.py
python scripts/check_no_llm_inference.py

Update docs/engineering/PROJECT_STATE.md before stopping.
```

---

# 22. 最终原则

你要把 Claude Code / Codex 当成“研究工程团队”，不是当成一个超长上下文聊天框。

```text
CLAUDE.md / AGENTS.md = 宪法
specs/*.md = 模块合同
skills = 可复用工作流
subagents = 隔离上下文的专业工人
Agent Teams = 有沟通需求的并行小队
Ralph = 有测试验收的长循环执行器
Superpowers = 工程纪律层
pytest / hooks / CI = 最终裁判
PROJECT_STATE.md = 跨会话记忆
```

对 GReaD-Core 来说，最安全的落地方式就是：

```text
先 Plan Mode 生成周全计划；
再用 Superpowers 规范执行；
再用 Ralph 跑单 Epic 长循环；
再用 subagents / Codex 审查；
最后用 tests、hooks、CI、paper-alignment gate 裁决。
```

**不要让任何 Agent 自由解释研究方案。让它们读 Markdown 契约、执行小任务、通过测试、更新状态。**

[1]: https://code.claude.com/docs/en/setup "Advanced setup - Claude Code Docs"
[2]: https://developers.openai.com/codex/cli "CLI – Codex | OpenAI Developers"
[3]: https://code.claude.com/docs/zh-CN/memory "Claude 如何记住你的项目 - Claude Code Docs"
[4]: https://claude.com/plugins/superpowers "Superpowers – Claude Plugin | Anthropic"
[5]: https://github.com/obra/superpowers "GitHub - obra/superpowers: An agentic skills framework & software development methodology that works. · GitHub"
[6]: https://claude.com/plugins/ralph-loop "Ralph Loop – Claude Plugin | Anthropic"
[7]: https://github.com/aezizhu/super-ralph "GitHub - aezizhu/super-ralph: Ralph autonomous loop + Superpowers skills = production-ready autonomous AI development. Based on obra/superpowers and frankbria/ralph-claude-code. · GitHub"
[8]: https://github.com/dollce/ultrawork "GitHub - dollce/ultrawork: Zero-learning-curve intelligent task orchestration with Hive-mind consensus · GitHub"
[9]: https://developers.openai.com/codex/skills "Agent Skills – Codex | OpenAI Developers"
[10]: https://code.claude.com/docs/zh-CN/sub-agents "创建自定义 subagents - Claude Code Docs"
[11]: https://code.claude.com/docs/en/agent-teams "Orchestrate teams of Claude Code sessions - Claude Code Docs"
[12]: https://claudecn.com/docs/claude-code/workflows/plan-mode/ "计划模式 – Claude 中文 - Claude AI 开发技术社区"
[13]: https://code.claude.com/docs/zh-CN/model-config "模型配置 - Claude Code Docs"
[14]: https://code.claude.com/docs/zh-CN/permissions "配置权限 - Claude Code Docs"
[15]: https://code.claude.com/docs/zh-CN/hooks "Hooks 参考 - Claude Code Docs"

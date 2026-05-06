下面为根据你上传内容整理润色后的**博士级标准研究方案版本**，在不改变原有研究方案核心设定、技术路线与贡献主张的基础上，对语句、结构、逻辑衔接与创新点表达进行了学术化重写。

---

# GReaD-Core：面向无 LLM 推理图欺诈检测的契约校验证据蒸馏框架

## 一、研究背景与问题动机

图欺诈检测是图异常检测与图深度学习领域中的重要研究方向，广泛应用于金融风控、电子商务、社交网络安全、恶意评论识别与垃圾行为检测等场景。与传统独立同分布数据不同，图欺诈检测面临节点属性、邻域结构、关系模式与欺诈行为策略之间高度耦合的问题。欺诈节点往往通过伪装邻居关系、模仿正常节点属性、制造局部结构异配性或形成高频异常信号来规避检测，因此，如何在保证检测性能的同时提供可验证、可解释、可部署的证据化推理能力，成为当前图欺诈检测研究中的关键挑战。

近年来，大语言模型在自然语言推理、解释生成与知识增强方面表现出较强能力，已有研究尝试将 LLM 引入图学习与图欺诈检测任务中，例如利用 LLM 提取文本语义、生成推理链或辅助 GNN 训练。然而，直接将完整图邻域、节点关系或自由文本解释交给 LLM 存在明显局限：其一，长上下文图结构输入容易带来拓扑幻觉与推理不稳定；其二，自由形式 rationale 难以保证证据闭合性与标签一致性；其三，在线调用 LLM 会显著增加推理成本，限制其在真实欺诈检测系统中的部署；其四，如果 LLM 能直接观察基础检测器的预测分数，则容易产生 score echo 或 label leakage，使生成解释退化为对已有分数的语言化复述。

因此，本研究提出 **GReaD-Core**，即 **Contract-Verified Score-Blind Evidence Distillation for LLM-Free Graph Fraud Reasoning**。该框架旨在将一个已经训练好的图欺诈检测器转化为具备证据化推理能力的轻量化模型，使其在推理阶段无需调用 LLM，仍能够同时输出欺诈分数、风险类型、有符号证据掩码与模板化解释。

---

## 二、研究目标与核心思想

GReaD-Core 的核心目标是构建一个**最小化、可验证、检测器原生、无在线 LLM 推理成本**的图欺诈检测推理框架。

其核心思想可以概括为：

> 先从已训练图欺诈检测器中抽取 score-blind 的 detector-native Minimal Evidence Package，再由 LLM teacher 基于该证据包生成结构化 Evidence Rationale Record；随后通过 Evidence Contract Verifier 对该 rationale 进行严格契约校验，仅将通过校验的 rationale 作为辅助监督蒸馏到轻量 student reasoner 中；最终在推理阶段完全移除 LLM，实现 LLM-free 的图欺诈检测与证据化解释。

该框架不主张将 LLM 作为在线图推理器，也不主张生成因果解释或提供任意检测器的通用保证。相反，GReaD-Core 的定位更加克制：它关注如何把 LLM 的结构化推理能力转化为**可验证、可蒸馏、可部署**的证据监督，从而增强图欺诈检测器的解释性与推理一致性。

---

## 三、问题定义

给定属性图：

[
G=(V,E,X)
]

其中 (V) 表示节点集合，(E) 表示边集合，(X) 表示节点属性矩阵。对于训练节点 (v \in V_{\text{train}})，其标签为：

[
y_v \in {0,1}
]

其中 (y_v=1) 表示欺诈节点，(y_v=0) 表示正常节点。

基础图欺诈检测器定义为：

[
f_\theta(G,X,v) \rightarrow (p_v,z_v)
]

其中，(p_v) 表示基础检测器输出的欺诈预测分数，(z_v) 表示节点 (v) 的表示向量。

GReaD-Core 不直接将完整图结构、邻居 ID、边 ID 或 k-hop 子图输入 LLM，而是通过检测器适配器抽取 score-blind 的证据表示：

[
A_d(f_\theta,G,X,v) \rightarrow E_v
]

其中 (E_v) 表示节点 (v) 的 Minimal Evidence Package。随后，LLM teacher 基于提示模板 (P(E_v)) 生成结构化 rationale：

[
M(P(E_v)) \rightarrow R_v
]

再由 Evidence Contract Verifier 判断该 rationale 是否满足预定义契约：

[
\text{ECV}(R_v,E_v,y_v) \rightarrow a_v \in {0,1}
]

只有当 (a_v=1) 时，生成的 Evidence Rationale Record 才会被用于后续 reasoning distillation。

---

## 四、总体框架设计

GReaD-Core 包含四个主要组成部分：

1. **Score-Blind Detector-Native Evidence Interface**
   从基础检测器中抽取与图欺诈风险相关的原生证据信号，同时隔离预测分数，避免 LLM 直接观察 fraud score。

2. **Evidence Rationale Record Generation**
   LLM teacher 仅基于 score-blind MEP 生成结构化 ERR，包括风险类型、支持证据、反向证据与简要解释。

3. **Evidence Contract Verifier**
   对 LLM 生成的 ERR 进行硬校验，确保其满足 schema validity、evidence availability、role consistency、risk-evidence contract、score-blindness 与 label compatibility。

4. **Evidence-Conditioned LLM-Free Student Reasoner**
   将通过校验的 rationale 蒸馏到轻量 student reasoner 中，使模型在无 LLM 推理阶段输出 fraud score、risk type、signed evidence mask 与 template explanation。

---

## 五、Score-Blind Minimal Evidence Package

为避免 prediction score 泄漏，GReaD-Core 将 MEP 设计为双通道结构：Calibration Channel 与 Reasoning Channel。该双通道结构在代码中通过 `MinimalEvidencePackage` 的 `to_teacher_payload()` 方法实现，该方法在构建 LLM prompt 前显式剥离 Calibration Channel，仅保留 Reasoning Channel。

### 5.1 Calibration Channel

Calibration Channel 仅用于 trace node selection、bucket 分层与实验分析，不进入 LLM reasoning prompt。其结构如下：

```json
{
  "prediction_score": 0.83,
  "uncertainty": 0.17
}
```

其中，`prediction_score` 为基础检测器输出的欺诈预测分数，`uncertainty = 1 - prediction_score` 表示预测不确定性。该通道保留基础检测器的校准信息，但在 `to_teacher_payload()` 调用时被完全剥离，确保 LLM teacher 在生成 rationale 时无法访问。

### 5.2 Reasoning Channel

Reasoning Channel 是 LLM 唯一可见的证据信息，其内容由离散化、结构化、可验证的 detector-native evidence 构成。每个证据字段采用离散语义等级（如 `low`、`medium`、`high`）或检测器原生信号标识符（如 `high_frequency_response_high`）表示，而非连续数值。这种离散化设计确保证据表示对 LLM 可读且可验证：

```json
{
  "uncertainty_level": "low",
  "degree_level": "high",
  "neighbor_consistency": "low",
  "feature_neighbor_discrepancy": "high",
  "detector_signal": "high_frequency_response_high",
  "detector_signal_strength": "strong",
  "counter_signal": "benign_neighbor_signal_medium",
  "allowed_support_ids": [
    "degree_level",
    "neighbor_consistency",
    "feature_neighbor_discrepancy",
    "detector_signal",
    "detector_signal_strength"
  ],
  "allowed_counter_ids": [
    "counter_signal",
    "uncertainty_level"
  ]
}
```

其中，`allowed_support_ids` 与 `allowed_counter_ids` 定义了该节点允许作为 supporting evidence 与 counter evidence 的字段集合。这些允许集合由 `ReasoningChannel` 的模型校验器（model validator）根据证据字段的语义角色自动生成。其核心约束包括：

1. **Score 隔离**：`prediction_score` 不提供给 LLM，且不得出现在 `allowed_support_ids` 或 `allowed_counter_ids` 中；
2. **角色互斥**：`counter_signal` 不得作为 supporting evidence；
3. **不确定性约束**：`uncertainty_level` 不能单独支撑强风险类型，只能支撑 `weak_or_uncertain_evidence` 或作为 counter evidence。

通过 score-blind 设计，LLM 无法基于基础检测器分数直接生成恶意或正常结论，而必须依赖 detector-native evidence 形成结构化 rationale。

---

## 六、Detector-Evidence Adapter Protocol

为提升方法的检测器适配性，GReaD-Core 将 evidence interface 抽象为统一协议。在代码实现中，该协议通过 `EvidenceAdapter` 基类定义，各检测器适配器（如 `GCNAdapter`、`BWGNNAdapter`、`CAREGNNAdapter`）继承该基类并实现 `extract()` 方法。证据包的结构化表示由 `MinimalEvidencePackage` 数据模型统一管理：

$$
E_v = E_{\text{generic}}(v) \cup E_{\text{detector}}(v) \cup E_{\text{counter}}(v)
$$

其中，$E_{\text{generic}}(v)$ 表示通用图证据，$E_{\text{detector}}(v)$ 表示检测器原生证据，$E_{\text{counter}}(v)$ 表示反向或不确定性证据。

### 6.1 Generic Evidence

通用证据不依赖具体检测器类型，可在多数图欺诈检测模型中通过图结构与节点特征计算得到：

| 证据字段 | 语义说明 |
| --- | --- |
| `degree_level` | 节点度数的离散化等级（如 `very_low`、`low`、`normal`、`high`、`burst`） |
| `neighbor_consistency` | 邻居标签或特征的一致性程度 |
| `feature_neighbor_discrepancy` | 节点属性与邻域聚合特征之间的差异程度 |
| `uncertainty_level` | 基于预测分数的不确定性估计（$1 - p_v$ 的离散化） |

### 6.2 Detector-Native Evidence

不同检测器提供不同的 detector-native signal，适配器负责将检测器内部状态转化为结构化证据字段：

| Base Detector | Detector-Native Evidence | 适用风险类型 |
| --- | --- | --- |
| BWGNN | normalized high-frequency / band-pass response | `spectral_anomaly` |
| CARE-GNN | camouflage-resistant neighbor selection score | `camouflage_neighbor` |
| GAT | attention concentration on suspicious neighbors | `structural_discrepancy` |
| GCN / GraphSAGE | message disagreement or embedding-neighbor discrepancy | `feature_structure_conflict` |
| XGBoost / LightGBM | feature importance risk signal | `structural_discrepancy` |

其中，BWGNN 的谱域高频响应适合作为 spectral anomaly 类型的检测器原生证据；CARE-GNN 中与伪装邻居过滤相关的选择信号适合作为 camouflage_neighbor 类型的证据；基于 attention 的模型可提供 suspicious neighbor concentration；传统树模型则可通过邻域聚合特征的重要性提供结构化风险证据。

### 6.3 Counter Evidence

反向证据字段用于避免模型只关注单向支持证据，使生成 rationale 能够显式区分 supporting evidence 与 counter evidence：

| 证据字段 | 语义说明 |
| --- | --- |
| `counter_signal` | 指向正常或低风险的反向信号（如 `benign_neighbor_signal_medium`） |
| `uncertainty_level` | 高不确定性本身构成对强风险判断的反向约束 |

在 `RiskTaxonomy` 中，`FORBIDDEN_SUPPORT_IDS` 显式定义了禁止作为 supporting evidence 的字段集合，包含 `prediction_score` 和 `counter_signal`。同时，`SCORE_RELATED_IDS` 定义了与预测分数相关的字段集合，用于 score-blindness 校验。

GReaD-Core 的 detector-adaptable 主张限定为：

> 当基础检测器能够暴露至少一个可计算的 detector-native evidence signal 时，GReaD-Core 可以通过 evidence adapter 将其转化为结构化证据接口。

该表述避免了对任意检测器的过强泛化承诺。

---

## 七、Trace Node Selection

为了构造高质量 LLM teacher supervision，本研究采用三桶 trace node selection 策略。在代码实现中，`TraceSelector` 类根据基础检测器的预测分数将候选节点划分为三个 bucket，每个 bucket 的预算比例可配置：

| Bucket | 选择标准 | 默认比例 |
| --- | --- | --- |
| `uncertain` | 预测分数接近 0.5 的节点 | 1/3 |
| `high_conf_fraud` | 预测分数接近 1.0 的节点 | 1/3 |
| `high_conf_benign` | 预测分数接近 0.0 的节点 | 1/3 |

在每个 bucket 内，进一步引入 evidence-pattern diversity selection。具体而言，首先将 MEP reasoning channel 的离散证据字段转化为向量表示，然后在每个 bucket 内采用多样性采样策略选择样本，使 trace nodes 尽可能覆盖多样化 evidence slot 组合。该多样性采样通过 `diversity_sampling` 配置项控制，默认启用。

该设计能够提升 LLM rationale 的证据覆盖范围，避免辅助监督集中于少数高频证据模式。

---

## 八、Evidence Rationale Record

LLM teacher 输出结构化 Evidence Rationale Record（ERR），而非自由文本解释。ERR 采用 signed evidence schema，在代码中通过 `EvidenceRationaleRecord` 数据模型定义：

```json
{
  "risk_type": "spectral_anomaly",
  "supporting_evidence": [
    "detector_signal",
    "neighbor_consistency"
  ],
  "counter_evidence": [
    "counter_signal"
  ],
  "summary": "The node shows strong high-frequency detector evidence and low neighbor consistency, while benign-neighbor evidence is only moderate."
}
```

ERR 包含四个字段：

| 字段 | 类型 | 训练用途 |
| --- | --- | --- |
| `risk_type` | `RiskType` 枚举值 | 作为风险类型分类目标 |
| `supporting_evidence` | 证据字段 ID 列表 | 作为正向证据掩码目标 $\hat{m}_v^+$ |
| `counter_evidence` | 证据字段 ID 列表 | 作为反向证据掩码目标 $\hat{m}_v^-$ |
| `summary` | 自由文本 | **不参与训练**，仅用于审计与分析 |

训练阶段仅使用 `risk_type`、`supporting_evidence` 与 `counter_evidence` 三个字段，通过 `training_targets()` 方法获取。`summary` 字段在任何情况下都不进入损失计算，该约束在 `ReasoningLoss` 与 `stage3_train_reasoner` 中均有显式保障。

其中，supporting evidence 与 counter evidence 分别对应正向证据掩码与反向证据掩码：

$$
\hat{m}_v^+, \hat{m}_v^-
$$

该设计避免将 counter signal 误作为 supporting evidence，同时使 student reasoner 能够学习不同证据角色之间的区别。

---

## 九、Risk Type Taxonomy

GReaD-Core 使用固定风险类型体系，在代码中通过 `RISK_TYPES_ORDERED` 定义，包含六种预定义风险类型：

| 风险类型 | 语义说明 |
| --- | --- |
| `structural_discrepancy` | 节点结构模式与正常节点存在明显差异 |
| `camouflage_neighbor` | 欺诈节点通过邻居伪装降低可检测性 |
| `spectral_anomaly` | 节点在检测器谱域或高频响应上表现异常 |
| `feature_structure_conflict` | 节点属性与邻域结构之间存在冲突 |
| `relation_or_burst_anomaly` | 关系类型或突发交互模式异常 |
| `weak_or_uncertain_evidence` | 当前证据不足以支撑强风险类型，或模型不确定性较高 |

每种风险类型在 Evidence Contract Verifier 中关联一组证据契约（详见第十节），定义该类型所需的必要证据条件与禁止证据条件。

与风险类型体系对应的证据槽位（evidence slots）通过 `EVIDENCE_SLOTS_ORDERED` 定义，包含七个标准证据字段：

```text
uncertainty_level, degree_level, neighbor_consistency,
feature_neighbor_discrepancy, detector_signal,
detector_signal_strength, counter_signal
```

在推理阶段，evidence slot 的索引位置与风险类型的索引位置分别用于构建证据掩码向量与类型分类目标，确保训练与推理的一致性。

在第一阶段实验中，可根据数据集特性激活其中 4–6 类。若以 BWGNN 为核心检测器，至少应覆盖 `structural_discrepancy`、`camouflage_neighbor`、`spectral_anomaly`、`feature_structure_conflict` 与 `weak_or_uncertain_evidence`。对于多关系图或时间突发数据集，可进一步启用 `relation_or_burst_anomaly`。

---

## 十、Evidence Contract Verifier

Evidence Contract Verifier 是 GReaD-Core 的关键模块，用于保证 LLM 生成的 rationale 不是仅满足格式要求的自由解释，而是满足严格证据契约的结构化监督。在代码实现中，`EvidenceContractVerifier` 类通过 `verify()` 方法依次执行六项校验，所有校验均通过后 ERR 才被接受。

Verifier 定义为：

$$
a_v = V_{\text{schema}} \cdot V_{\text{availability}} \cdot V_{\text{role}} \cdot V_{\text{contract}} \cdot V_{\text{score-blind}} \cdot V_{\text{label}}
$$

只有当所有校验项均通过时，ERR 才会被接受（$a_v=1$）。任一校验失败，`VerificationResult.accepted` 返回 `False`，并附带具体失败原因列表。

### 10.1 Schema Validity

Schema 校验确保 ERR 满足基本结构要求：

$$
V_{\text{schema}} = \mathbb{1}[\text{valid JSON}] \cdot \mathbb{1}[t_v \in \mathcal{T}] \cdot \mathbb{1}[\text{supporting\_evidence} \in \text{List}] \cdot \mathbb{1}[\text{counter\_evidence} \in \text{List}]
$$

其中 $\mathcal{T}$ 为固定风险类型集合。该校验通过 Pydantic 模型的自动验证实现，`EvidenceRationaleRecord` 定义了 `risk_type` 为 `RiskType` 枚举类型，`supporting_evidence` 与 `counter_evidence` 均为 `list[str]` 类型。

### 10.2 Evidence Availability

证据可用性校验确保 ERR 引用的所有证据字段均存在于 MEP Reasoning Channel 中：

$$
V_{\text{availability}} = \mathbb{1}[\forall e \in \text{supporting\_evidence} \cup \text{counter\_evidence}: e \in E_v]
$$

当某一证据字段值为 `unavailable` 时，该校验确保该字段不得被引用。例如，若 MEP 中 `"detector_signal": "unavailable"`，则 `detector_signal` 不得出现在 supporting evidence 或 counter evidence 中。

### 10.3 Role Consistency

角色一致性校验确保 supporting evidence 与 counter evidence 分别落在允许的字段集合内，且两者不重叠：

$$
V_{\text{role}} = \mathbb{1}[\text{supporting\_evidence} \subseteq \text{allowed\_support\_ids}] \cdot \mathbb{1}[\text{counter\_evidence} \subseteq \text{allowed\_counter\_ids}] \cdot \mathbb{1}[\text{supporting\_evidence} \cap \text{counter\_evidence} = \emptyset]
$$

`allowed_support_ids` 与 `allowed_counter_ids` 由 `ReasoningChannel` 的模型校验器根据证据字段语义角色自动生成，其中 `FORBIDDEN_SUPPORT_IDS`（包含 `prediction_score` 与 `counter_signal`）中的字段不会出现在 `allowed_support_ids` 中。

### 10.4 Risk-Evidence Contract

对每个 risk type 定义证据契约：

$$
C_t = (R_t, O_t, F_t)
$$

其中 $R_t$ 为必要证据条件，$O_t$ 为可选证据条件，$F_t$ 为禁止证据条件。

**关键实现细节**：代码中 `required_any` 条件的满足采用**双重判定机制**——一个必要条件被满足当且仅当同时满足以下两个子条件：

1. **值匹配**（value match）：MEP 中对应字段的值落入契约规定的值集合；
2. **证据引用**（evidence citation）：该字段被 ERR 的 `supporting_evidence` 引用。

形式化地，对于必要条件 $c = (\text{field}, \text{values})$：

$$
\text{satisfied}(c) = \mathbb{1}[\text{reasoning}[\text{field}] \in \text{values}] \cdot \mathbb{1}[\text{field} \in \text{supporting\_evidence}]
$$

该双重判定机制确保 LLM 不仅生成了语义上正确的证据值，而且在 ERR 中显式引用了该证据字段作为支撑依据，避免"合规但未引用"的隐性漏洞。

各风险类型的证据契约示例如下：

**spectral_anomaly**：

```text
Required (any):
  detector_signal ∈ {high_frequency_response_high, spectral_energy_shift_high, bandpass_response_high}
  — 且该字段必须被 supporting_evidence 引用

Forbidden:
  detector_signal = unavailable
  detector_signal_strength = weak
```

**camouflage_neighbor**：

```text
Required (any):
  neighbor_consistency = low — 且被 supporting_evidence 引用
  detector_signal = camouflage_neighbor_filter_high — 且被 supporting_evidence 引用

Forbidden:
  neighbor_consistency = high
```

**feature_structure_conflict**：

```text
Required (any):
  feature_neighbor_discrepancy = high — 且被 supporting_evidence 引用

Forbidden:
  feature_neighbor_discrepancy ∈ {low, unavailable}
```

**structural_discrepancy**：

```text
Required (any):
  degree_level ∈ {very_low, high, burst} — 且被 supporting_evidence 引用
  neighbor_consistency = low — 且被 supporting_evidence 引用

Forbidden:
  degree_level = normal
```

**relation_or_burst_anomaly**：

```text
Required (any):
  degree_level = burst — 且被 supporting_evidence 引用

Forbidden:
  degree_level = normal AND neighbor_consistency = high
```

**weak_or_uncertain_evidence**：

```text
Required (any):
  uncertainty_level = high — 且被 supporting_evidence 引用
  detector_signal = unavailable — 且被 supporting_evidence 引用

Forbidden:
  detector_signal ∈ {high_frequency_response_high, ...} AND detector_signal_strength = strong
```

### 10.5 Score-Blindness Check

Score-blindness 校验拒绝任何将 prediction score 或其相关字段作为证据的 ERR。在代码实现中，该校验通过 `SCORE_RELATED_IDS` 集合定义与预测分数相关的所有字段 ID，并检查这些字段是否出现在 ERR 的 supporting evidence 或 counter evidence 中：

$$
V_{\text{score-blind}} = \mathbb{1}[(\text{supporting\_evidence} \cup \text{counter\_evidence}) \cap \text{SCORE\_RELATED\_IDS} = \emptyset]
$$

该校验可通过 `score_blind` 配置项控制开关。在主方法中默认启用（`score_blind=True`）；在消融实验中可关闭以评估 score-blind 设计的必要性。

### 10.6 Label Compatibility Check

对于训练集中有标签的 trace nodes，引入标签-风险类型兼容性约束。在代码实现中，该约束通过可配置的禁止类型列表实现：

$$
V_{\text{label}} =
\begin{cases}
\mathbb{1}[t_v \notin \mathcal{F}_{\text{fraud}}] & \text{if } y_v = 1 \\
\mathbb{1}[t_v \notin \mathcal{F}_{\text{benign}}] & \text{if } y_v = 0 \\
1 & \text{if } y_v = \text{None}
\end{cases}
$$

其中 $\mathcal{F}_{\text{fraud}}$ 与 $\mathcal{F}_{\text{benign}}$ 分别为欺诈标签与正常标签对应的禁止风险类型列表，可通过合约配置文件自定义。该校验可通过 `label_compatibility.enabled` 配置项控制开关。

该校验防止"合规但事实错误"的 LLM rationale 被蒸馏进模型。例如，若 $y_v=1$（欺诈节点），则 risk type 不应被标记为纯粹 benign 类型；若 $y_v=0$（正常节点），则不应生成强恶意风险类型。

### 10.7 Verifier Soundness Statement

GReaD-Core 可形成如下克制而明确的性质表述：

> 若 $\text{ECV}(R_v, E_v, y_v) = 1$，则被接受的 ERR 在预定义风险类型体系下满足 schema-valid、evidence-closed、role-consistent、score-blind、contract-consistent 与 label-compatible。

该性质不声称解释具有因果正确性，而是强调其满足契约一致性与可扰动检验条件。该 soundness 保证的严格性源于六项校验的串联结构：任一校验失败即拒绝，不存在"部分通过"的中间状态。

---

## 十一、Evidence-Conditioned Student Reasoner

原始并行 head 设计容易导致解释输出与最终 fraud score 脱耦。为提升 evidence responsiveness，GReaD-Core 引入 evidence-conditioned student reasoner，在代码中通过 `GReaDReasoner` 类实现。该类协调四个子模块：`EvidenceEncoder`、`RiskTypeHead`、`SignedEvidenceHead` 与 `EvidenceGatedResidualReadout`。

### 11.1 Evidence Encoding

首先对 MEP Reasoning Channel 进行离散化编码。在代码实现中，`EvidenceEncoder` 将证据字段名映射为离散 token ID（slot $i$ 对应 token $i+1$，token 0 为 padding），然后通过 Embedding 层与投影层生成稠密证据表示：

$$
g_v = \phi(E_v) = \text{LayerNorm}(\text{ReLU}(\mathbf{W}_\phi \cdot \text{flatten}(\text{Embedding}(\text{token\_ids}))))
$$

具体而言，证据 token ID 序列 $[B, K]$ 经 Embedding 层映射为 $[B, K, d_e]$，展平为 $[B, K \cdot d_e]$ 后通过线性投影与非线性激活输出 $[B, E]$ 维的证据嵌入 $g_v$。该设计采用离散 token 表示而非连续数值编码，确保证据表示与 LLM prompt 中的离散语义等级对齐。

### 11.2 Risk Type Prediction

将节点表示与证据表示拼接，用于预测风险类型：

$$
\hat{t}_v = h_{\text{type}}([z_v; g_v]) = \mathbf{W}_t [z_v; g_v] + \mathbf{b}_t
$$

其中 $h_{\text{type}}$ 为单层线性投影（`RiskTypeHead`），输出维度为风险类型数量 $T$。$\hat{t}_v \in \mathbb{R}^T$ 为风险类型 logits，推理时通过 $\arg\max$ 获取预测风险类型。

### 11.3 Signed Evidence Mask Prediction

有符号证据掩码通过两个**独立的线性投影头**（`SignedEvidenceHead`）分别预测正向与反向证据掩码：

$$
\hat{m}_v^+ = \sigma(\mathbf{W}_+ [z_v; g_v] + \mathbf{b}_+)
$$

$$
\hat{m}_v^- = \sigma(\mathbf{W}_- [z_v; g_v] + \mathbf{b}_-)
$$

其中 $\mathbf{W}_+$ 与 $\mathbf{W}_-$ 为两个**不共享参数**的独立线性层，允许模型分别学习 supporting evidence 与 counter evidence 的不同模式。输出维度均为证据槽位数量 $K$。

### 11.4 Evidence-Gated Residual Fraud Readout

为使 evidence mask 对最终 fraud score 具有实际影响，引入 evidence-gated residual readout。在代码实现中，`EvidenceGatedResidualReadout` 模块的计算分为两步：

**第一步**：通过 MLP 计算原始残差：

$$
r_v^{\text{raw}} = \text{MLP}([z_v; g_v]) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 [z_v; g_v] + \mathbf{b}_1) + \mathbf{b}_2
$$

其中 MLP 为两层前馈网络，输入维度为 $H + E$，输出维度为 1。

**第二步**：计算证据门控系数：

$$
\alpha_v = \text{mean}(\sigma(\hat{m}_v^+)) - \text{mean}(\sigma(\hat{m}_v^-))
$$

其中 $\sigma$ 为 sigmoid 函数，$\text{mean}(\cdot)$ 对证据槽位维度取均值。$\alpha_v$ 为标量，表示正向证据与反向证据的净方向。

**最终残差**：

$$
r_v = r_v^{\text{raw}} \cdot \alpha_v
$$

最终欺诈分数为：

$$
\text{logit}(s_v) = \text{logit}(p_v^{\text{base}}) + \rho \cdot r_v
$$

其中 $\rho$ 为残差缩放系数（默认 0.1），通过配置项 `residual_rho` 控制。

**与公式表达的差异说明**：该实现与直观公式 $r_v = q_\psi([z_v; g_v \odot (\hat{m}_v^+ - \hat{m}_v^-)])$ 存在设计差异。代码采用"MLP 先行、门控后置"的架构——先在完整 $[z_v; g_v]$ 上计算残差方向，再通过证据掩码的净 sigmoid 均值差进行标量调制。该设计的优势在于：（1）MLP 可以在不受掩码噪声干扰的情况下学习残差方向；（2）门控系数 $\alpha_v$ 具有明确的概率解释（正向证据激活度减去反向证据激活度）；（3）当所有证据掩码趋近 0.5 时，$\alpha_v \to 0$，残差贡献自然衰减。

该设计保留基础检测器作为主预测器，同时允许证据推理模块对最终分数进行轻量残差校准，使模型预测在受控证据扰动下表现出可检验的响应性。

---

## 十二、训练流程

GReaD-Core 采用三阶段训练协议，在代码中分别通过 `train_detector`、`generate_err` 与 `train_reasoner` 三个 CLI 模块实现。

### 12.1 Stage 1：Base Detector Warm-up

首先训练基础图欺诈检测器：

$$
\min_\theta \mathcal{L}_{\text{sup}}(p_v, y_v)
$$

该阶段得到稳定的预测分数 $p_v$、节点表示 $z_v$ 以及各检测器原生信号。该阶段保证后续 MEP 抽取基于已经具备判别能力的检测器，避免在训练早期由噪声检测器生成低质量证据包。

### 12.2 Stage 2：Offline ERR Generation and Verification

对 trace nodes 抽取 score-blind MEP：

$$
E_v = A_d(f_\theta, G, X, v)
$$

调用 LLM teacher 生成 ERR：

$$
R_v = M(P(E_v))
$$

随后通过 Evidence Contract Verifier：

$$
a_v = \text{ECV}(R_v, E_v, y_v)
$$

仅保留 $a_v=1$ 的 ERR 作为蒸馏监督。在代码实现中，Stage 2 通过 `Stage2Runtime` 管理 LLM 后端选择（`stub`、`replay`、`openai`），并通过 `PromptCache` 实现 LLM 响应的缓存与复用，避免重复调用。

### 12.3 Stage 3：Reasoner Distillation

Stage 3 冻结基础检测器参数，仅训练 evidence-conditioned student reasoner。在代码实现中，`ReasoningLoss` 类封装了完整的损失计算逻辑。

**监督检测损失** $\mathcal{L}_{\text{sup}}$ 在所有训练样本上计算，采用基于 BCE 的监督损失函数：

$$
\mathcal{L}_{\text{sup}} = \text{supervised\_loss}(\text{logit}(s_v), y_v)
$$

**风险类型损失** $\mathcal{L}_{\text{type}}$ 仅在已接受 ERR 的样本上计算，采用逐样本交叉熵损失：

$$
\mathcal{L}_{\text{type}} = \frac{1}{|\mathcal{A}|} \sum_{v \in \mathcal{A}} \text{CE}(\hat{t}_v, t_v)
$$

其中 $\mathcal{A} = \{v : a_v = 1\}$ 为通过契约校验的样本集合，$t_v$ 为 ERR 中的风险类型标签。

**有符号证据损失** $\mathcal{L}_{\text{evidence}}$ 仅在已接受 ERR 的样本上计算，采用逐样本 BCE 损失：

$$
\mathcal{L}_{\text{evidence}} = \frac{1}{|\mathcal{A}|} \sum_{v \in \mathcal{A}} \left[ \text{BCE}(\hat{m}_v^+, m_v^+) + \text{BCE}(\hat{m}_v^-, m_v^-) \right]
$$

其中 $m_v^+$ 与 $m_v^-$ 分别为由 ERR 的 `supporting_evidence` 与 `counter_evidence` 字段编码得到的二值目标向量。

**总损失**：

$$
\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda \cdot (\mathcal{L}_{\text{type}} + \mathcal{L}_{\text{evidence}})
$$

其中 $\lambda$ 为推理蒸馏损失权重（`lambda_reason`，默认 0.5）。

**关于 $a_v$ 的实现说明**：在代码实现中，$a_v$ 以布尔掩码 `accepted_mask` 的形式参与损失计算。已接受样本（$a_v=1$）的类型损失与证据损失被纳入梯度更新；被拒绝样本（$a_v=0$）的类型损失与证据损失为零，不贡献梯度。若当前 batch 中无已接受样本（`accepted_mask.sum() == 0`），则类型损失与证据损失均返回 0，总损失退化为 $\mathcal{L}_{\text{sup}}$。该机制确保被拒绝的"合规但错误"的 ERR 不会污染推理蒸馏过程。

主训练目标保持为：

$$
\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda \cdot a_v \cdot (\mathcal{L}_{\text{type}} + \mathcal{L}_{\text{evidence}})
$$

---

## 十三、推理阶段

推理阶段完全不调用 LLM，在代码中通过 `GReaDInferencePipeline` 类实现。该类协调基础检测器、证据适配器与 student reasoner，完成端到端的无 LLM 推理。

对于输入节点 $v$，推理流程如下：

1. **检测器前向传播**：调用 `detector.forward_with_embedding(graph)` 获取全部节点的 base logit 与节点表示；
2. **证据抽取**：调用 `adapter.extract(node_ids)` 获取各节点的 MEP；
3. **证据编码**：将 MEP Reasoning Channel 的字段名映射为离散 token ID，构建 `evidence_token_ids` 张量；
4. **Reasoner 前向传播**：调用 `reasoner.forward(z_v, base_logit, evidence_token_ids)` 获取 `final_logit`、`type_logits` 与证据掩码 logits；
5. **结果组装**：风险类型由 $\arg\max(\text{type\_logits})$ 确定；supporting evidence 由正向掩码 logits 经阈值（默认 0.5）二值化后映射为证据字段名；counter evidence 由反向掩码 logits 同理得到。

Student reasoner 输出结构如下：

```json
{
  "fraud_score": 0.84,
  "risk_type": "spectral_anomaly",
  "supporting_evidence": ["detector_signal", "neighbor_consistency"],
  "counter_evidence": ["counter_signal"],
  "explanation": "The node is classified as spectral_anomaly based on supporting evidence [detector_signal, neighbor_consistency] and counter evidence [counter_signal]."
}
```

**解释生成**：最终解释由确定性模板生成，而非在线调用 LLM。在代码实现中，`generate_explanation()` 函数采用简单字符串格式化：

$$
\text{explanation} = \texttt{"The node is classified as \{risk\_type\} based on supporting evidence [\{support\}] and counter evidence [\{counter\}]"}
$$

该模板化设计保证解释输出的确定性、可复现性与可控性，避免 LLM 生成的随机性与不可控性。因此，GReaD-Core 在部署阶段具备低成本、稳定输出与可控解释格式的优势。

---

## 十四、实验设计

实验评估分为三层：检测性能、推理质量与解释响应性。

### 14.1 Detection Performance

报告以下指标：

```text
ROC-AUC
AUPRC
Recall@K
F1
Precision@K
```

考虑到欺诈检测通常是类别不平衡任务，AUPRC、Recall@K 与 Precision@K 应作为重点指标。

### 14.2 Reasoning Quality

评估结构化推理输出质量：

```text
Verifier acceptance rate
Contract violation rate
Risk-type agreement on accepted ERR
Evidence F1 against accepted ERR
Evidence sparsity
Signed evidence role accuracy
Template validity
```

若数据集中不存在人工标注的 risk type ground truth，可采用专家抽样审计：

```text
human audit on 200 sampled rationales
expert agreement
LLM-as-judge only as auxiliary, not as ground truth
```

### 14.3 Faithfulness / Responsiveness Evaluation

将 Counterfactual Evidence Consistency 扩展为三元指标。

#### CEC-score

[
CEC_{\text{score}}
==================

\frac{1}{N}
\sum_v
\mathbb{1}
[
s_v(E_v)-s_v(E_v^{-})>0
]
]

#### CEC-type

[
CEC_{\text{type}}
=================

\frac{1}{N}
\sum_v
\mathbb{1}
[
P(\hat t_v=t_v|E_v)
-------------------

P(\hat t_v=t_v|E_v^{-})

> 0
> ]
> ]

#### CEC-evidence

[
CEC_{\text{evidence}}
=====================

\frac{1}{N}
\sum_v
\mathbb{1}
[
P(\hat m_v^+|E_v)
-----------------

P(\hat m_v^+|E_v^{-})

> 0
> ]
> ]

其中，(E_v^{-}) 表示削弱 supporting evidence 后的 MEP。例如：

```text
detector_signal = high_frequency_response_high
→ detector_signal = neutral
```

或：

```text
neighbor_consistency = low
→ neighbor_consistency = medium/high
```

该评价不声称因果解释保证，而是检验模型输出是否具有 counterfactual evidence responsiveness。

---

## 十五、Non-Redundancy Test

为证明 GReaD-Core 学到的 reasoning output 不是 base score 的简单复述，需要进行 non-redundancy test。

定义：

```text
Y = ground-truth fraud label
P = base prediction score
T = predicted risk type
M = predicted evidence mask
```

目标是验证：

[
I(T,M;Y|P)>0
]

实验上可采用替代检验：

```text
AUC(Y ~ P)
AUC(Y ~ P + T)
AUC(Y ~ P + T + M)
```

以及：

```text
AUPRC(Y ~ P)
AUPRC(Y ~ P + T)
AUPRC(Y ~ P + T + M)
```

若加入 (T) 与 (M) 后能够带来稳定增益，则说明 reasoning outputs 对 ground-truth label 具有超越 base score 的条件增量信息。

同时报告：

```text
corr(prediction_score, risk_type_confidence)
corr(prediction_score, evidence_mask_confidence)
```

理想情况下，risk type confidence 与 evidence mask confidence 不应与 prediction score 呈现过高相关性，否则可能表明 reasoning module 存在 score echo。

---

## 十六、对比方法与消融实验

### 16.1 Baselines

实验中应至少包含以下对比方法：

```text
Base detector only
Base detector + naive heads without LLM
Base detector + LLM ERR without verifier
Base detector + schema verifier only
Base detector + Evidence Contract Verifier
GNNExplainer / PGExplainer-style explanation baseline
MLED / FraudCoT if dataset has text-attributed graph setting
Tree ensemble + neighborhood aggregation
```

其中，tree ensemble + neighborhood aggregation 是必要强基线，用于验证 GReaD-Core 的增益并非来自简单邻域特征建模。

### 16.2 Ablations

关键消融实验包括：

```text
w/o score-blind MEP
w/o Evidence Contract Verifier
w/o label compatibility
w/o role consistency
w/o evidence-gated residual readout
w/o signed evidence mask
w/o diversity trace selection
fixed λ vs warm-up λ
BWGNN only vs multi-detector adapter
```

消融重点应围绕以下四组展开：

1. `schema verifier only` vs `Evidence Contract Verifier`
2. `score-visible MEP` vs `score-blind MEP`
3. `parallel heads` vs `evidence-gated residual readout`
4. `single detector` vs `adapter protocol across detectors`

---

## 十七、核心创新点与贡献总结

本文最终核心贡献凝练为三点。

### Contribution 1：Score-Blind Detector-Native Evidence Interface

本文提出 score-blind 的 detector-native evidence interface，将图欺诈检测器内部的结构、属性、邻域一致性与检测器原生信号转化为 LLM 可读、可验证、可蒸馏的 Minimal Evidence Package。MEP 采用双通道结构（Calibration Channel 与 Reasoning Channel），通过 `to_teacher_payload()` 方法在构建 LLM prompt 前显式剥离 Calibration Channel。与直接将预测分数或完整图邻域输入 LLM 不同，该接口将 prediction score 从 reasoning evidence 中隔离，降低 score leakage 与 rationale echo 风险，使 LLM teacher 必须基于结构化证据生成推理记录。

### Contribution 2：Contract-Verified Reasoning Distillation

本文提出 Evidence Contract Verifier，对 LLM 生成的 Evidence Rationale Record 进行六项严格契约校验：schema validity、evidence availability、role consistency、risk-evidence contract、score-blindness 与 label compatibility。其中，risk-evidence contract 校验采用双重判定机制——必要条件的满足不仅要求 MEP 字段值匹配契约值集合，还要求该字段被 ERR 的 supporting_evidence 显式引用。只有通过全部校验的 rationale 才能作为辅助监督进入蒸馏，从而提高 LLM-generated reasoning supervision 的可靠性、闭合性与训练可控性。

### Contribution 3：Evidence-Conditioned LLM-Free Reasoner

本文提出 evidence-conditioned student reasoner，使模型在推理阶段无需调用 LLM，即可输出 fraud score、risk type、signed evidence mask 与 template explanation。该 reasoner 由四个子模块组成：EvidenceEncoder（离散 token Embedding + MLP 投影）、RiskTypeHead（线性分类头）、SignedEvidenceHead（两个独立线性头分别预测正向与反向证据掩码）与 EvidenceGatedResidualReadout（MLP 残差 + 证据门控）。通过 evidence-gated residual readout，证据掩码不再只是独立解释输出，而是通过标量门控系数参与最终预测分数的轻量残差校准。进一步地，本文通过 tri-level Counterfactual Evidence Consistency 与 non-redundancy test 评估解释输出的证据响应性及其相对于 base detector score 的增量价值。

---

## 十八、摘要草案

### 中文摘要

本文提出 GReaD-Core，一个面向图欺诈检测的契约校验、分数盲化证据蒸馏框架。不同于直接将完整图邻域输入大语言模型，或将自由文本解释直接作为伪监督的方法，GReaD-Core 首先将已训练图欺诈检测器转化为紧凑的 detector-native evidence interface，采用双通道 Minimal Evidence Package 结构，通过 `to_teacher_payload()` 在构建 LLM prompt 前显式剥离校准通道。LLM teacher 仅基于 score-blind 的证据字段生成结构化 Evidence Rationale Record，包含风险类型、支持证据、反向证据与摘要。生成的 rationale 需要通过 Evidence Contract Verifier 的六项严格校验（schema validity、evidence availability、role consistency、risk-evidence contract with dual-condition check、score-blindness 与 label compatibility），只有被接受的 rationale 才会进入 reasoning distillation。推理阶段，GReaD-Core 不需要调用 LLM，即可输出欺诈分数、风险类型、有符号证据掩码与模板化解释。进一步地，本文通过三元 Counterfactual Evidence Consistency 与 non-redundancy test 验证模型解释对证据扰动的响应性，以及 reasoning output 相对于基础检测器分数的增量价值。

### 英文摘要

We propose GReaD-Core, a contract-verified score-blind evidence distillation framework for graph fraud detection. Instead of prompting large language models with full graph neighborhoods or treating free-form rationales as pseudo-labels, GReaD-Core first converts a trained graph fraud detector into a compact detector-native evidence interface with a dual-channel Minimal Evidence Package, explicitly stripping the calibration channel before constructing the LLM prompt via `to_teacher_payload()`. The LLM teacher only observes score-blind evidence fields and generates a structured Evidence Rationale Record containing risk type, supporting evidence, counter evidence, and summary over a fixed risk taxonomy. The generated rationale is accepted only if it passes all six checks of an Evidence Contract Verifier: schema validity, evidence availability, evidence role consistency, risk-evidence contract with dual-condition verification, score-blindness, and label compatibility. Accepted rationales supervise a lightweight evidence-conditioned reasoner comprising an EvidenceEncoder (discrete token embedding with MLP projection), a RiskTypeHead (linear classifier), a SignedEvidenceHead (two independent linear heads for supporting and counter evidence masks), and an EvidenceGatedResidualReadout (MLP residual with scalar evidence gate) through a single auxiliary reasoning loss. At inference time, GReaD-Core requires no LLM calls and outputs fraud scores, risk types, signed evidence masks, and deterministic template-based explanations. We further evaluate explanation responsiveness using tri-level Counterfactual Evidence Consistency and demonstrate that the learned reasoning outputs provide non-redundant information beyond the base detector score.

---

## 十九、一句话总结

**GReaD-Core 通过 score-blind 双通道 MEP、六项契约校验（含双重判定机制）与 evidence-gated residual reasoner（独立有符号证据头 + 标量门控残差），将 LLM 的结构化推理能力蒸馏为可验证、可部署、无在线 LLM 成本的图欺诈检测证据化推理能力。**

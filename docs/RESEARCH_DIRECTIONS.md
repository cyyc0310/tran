# TransCIF 顶会投稿：研究方向与技术路线

> 目标：基于当前 config-only zero-shot 碳排放强度预测框架，在 2026-2027 投稿周期内向 NeurIPS / ICML / ICLR / KDD 等顶会投稿。
> 核心策略：结构创新 + 跨域方法迁移，充分利用物理层的独特优势作为差异化竞争力。
> 当前日期：2026 年 8 月 4 日。最近的主投稿窗口为 ICLR 2027（约 2026.10）、ICML 2027（约 2027.01）、NeurIPS 2027（约 2027.05）。

---

## 目录

1. [当前项目的核心竞争力盘点](#一当前项目的核心竞争力盘点)
2. [方向一：检索增强零样本时序预测（RAG-TS）](#二方向一检索增强零样本时序预测)
3. [方向二：物理约束的不变风险最小化（Phys-IRM）](#三方向二物理约束的不变风险最小化)
4. [方向三：反事实域解耦与因果自适应（Causal-ZS）](#四方向三反事实域解耦与因果自适应)
5. [方向四：上下文学习器用于时序预测（IC-TSF）](#五方向四上下文学习器用于时序预测)
6. [方向五：去偏一致性分层预测（Debiased-Hier）](#六方向五去偏一致性分层预测)
7. [方向总览与投稿策略](#七方向总览与投稿策略)
8. [短期行动计划](#八短期行动计划)

---

## 一、当前项目的核心竞争力盘点

在讨论具体研究方向之前，先梳理本项目已有的差异化优势，这些是任何顶会投稿都需要充分利用的：

| 优势 | 具体内容 | 审稿人视角 |
|---|---|---|
| **独特的问题定义** | Config-only zero-shot：只给两个物理标量 + 实时 RenewShare，预测新区域 CIF | "This is a real-world problem with clear practical value" |
| **可验证的物理层** | CIF = s·e_r + (1-s)·e_nr，误差可精确分解 | "The physics decomposition enables rigorous ablation" |
| **严格的定理体系** | Theorem 1（误差传播上界）+ Theorem 2（迁移难度 U 型曲线） | "The theoretical analysis is non-trivial" |
| **大规模实证** | 29 区域 × 3 司法管辖区 × 5 种子 LORO | "The empirical scale is convincing" |
| **低参数基线** | 18K 参数模型已经具有竞争力 | "Not just scaling up→ reasonable baseline" |

---

## 二、方向一：检索增强零样本时序预测

### 2.1 核心思想

**将 NLP 领域的 RAG（Retrieval-Augmented Generation）范式第一次系统性地引入长时间序列预测任务。**

具体而言：

- **检索库**：所有源区域的时间窗口，按配置向量索引；
- **检索策略**：给定目标区域的配置和当前输入窗口，检索 top-k 最相关的源区域历史片段；
- **增强生成**：将检索到的片段作为附加条件，与目标输入一起送入预测器；
- **物理重建**：检索增强的 RenewShare 预测，经过物理层转换为 CIF。

与 NLP RAG 的本质差异：
- 检索单元不是文本块，而是**多元时间片段**；
- 相似度不是语义相似度，而是**配置距离 + 动态模式距离**；
- 生成器不是 LLM，而是**物理约束的轻量时序预测器**。

### 2.2 为什么这个方向适合顶会

1. **RAG 概念极热但时序领域几乎空白**。现有的 TimeRAG（arxiv 2412.16643）和 TS-RAG（arxiv 2503.07649）都基于 LLM 的文本化时序表示，需要将时序 token 化、构建提示词。*非 LLM 的纯时序 RAG 尚无人提出*。

2. **物理层使 RAG 变得优雅**：检索增强的是 Renewable Share 的预测，物理层自然处理 CIF 转换。检索到的序列即使来自不同排放因子区域，其 share 模式仍可迁移——物理层自动适配目标区域的因子。

3. **可解释性极强**：每个预测可以追溯到具体的检索源，审稿人可以直接看到"这个预测受 QLD 第 3 天模式影响"。

4. **消融实验丰富**：检索 vs 不检索、不同相似度度量、不同 top-k、不同检索库构建策略、检索 vs 加权平均。

### 2.3 技术方案

#### 2.3.1 检索库构建

```
Memory Bank M = {(c_i, x_i^{t-L:t}, y_i^{t:t+H})}
for each source region i and each valid time window t
where c_i = (mean_rs_i, ef_nr_i) is the config vector
```

每个条目同时存储配置向量和完整的 (输入, 输出) 对。

#### 2.3.2 检索相似度

两步检索：

1. **粗筛（配置距离）**：
   ```
   sim_config(c_target, c_i) = -||c_target - c_i||_2
   ```

2. **精排（动态模式距离）**：
   对粗筛后的候选，计算输入窗口的模式相似度：
   ```
   sim_pattern(x_target, x_i) = DTW(x_target, x_i)    # 动态时间规整
   ```
   或使用 learnable pattern encoder 将窗口嵌入后计算余弦相似度。

3. **联合相似度**：
   ```
   sim_total = α·sim_config + (1-α)·sim_pattern
   ```

#### 2.3.3 检索增强预测器

```
输入: [x_target; x_retrieved_1; x_retrieved_2; ...; x_retrieved_k]
           ↓
   Cross-Attention Encoder
   x_target 作为 query, retrieved top-k 作为 key & value
           ↓
   Prediction Head → ŝ
           ↓
   Physics Layer → CIF
```

可选方案：

- **方案 A（简洁）**：将检索到的 top-k 序列与目标序列拼接，送入 DLinear-style 模型；
- **方案 B（优雅）**：使用 cross-attention，目标序列 attend 到检索序列上；
- **方案 C（最强）**：每个检索片段的 RenewShare 先独立预测，再通过 learnable gating 与目标预测融合。

#### 2.3.4 训练策略

- **端到端训练**：在源区域上进行 LORO 训练，每次留一个区域时，检索库只包含其余区域；
- **对比学习辅助任务**：正样本是配置最近区域的相似窗口，负样本是配置最远区域的窗口；
- **梯度分离**：检索器的相似度计算可以独立训练，也可以与预测器联合优化。

### 2.4 创新空间与发表可行性

| 维度 | 评估 |
|---|---|
| **方法新颖度** | ★★★★★ RAG 在时序领域处于非常早期阶段 |
| **技术深度** | ★★★★☆ 需要设计检索策略、attention 机制、训练方案 |
| **与现有工作区分** | ★★★★★ 所有现有 RAG-TS 都是 LLM-based，纯时序 RAG 是空白 |
| **实验充分性** | ★★★★★ 29 区域 + 消融 + 案例分析 |
| **目标会议** | NeurIPS / ICML |
| **审稿风险** | "Why not just use weighted training?" — 需在 intro 中直接回答 |
| **发表可行性** | **高**（核心概念新颖、与物理层协同自然、实验体量大） |

### 2.5 关键文献

- TimeRAG: Boosting LLM Time Series Forecasting via Retrieval-Augmented Generation, arxiv 2412.16643
- TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster, arxiv 2503.07649
- RAG 基础：Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020
- 对比学习：(Chen et al., SimCLR; He et al., MoCo)

---

## 三、方向二：物理约束的不变风险最小化

### 3.1 核心思想

**将 IRM（Invariant Risk Minimization）框架与物理误差分解定理结合，提出一种具有理论保证的域泛化训练策略。**

核心观察（来自 Theorem 1）：

\[
\underbrace{\text{CIF error}}_{\text{final loss}}
=
\underbrace{|C_{\text{renew}} - C_{\text{nonrenew}}| \cdot |\hat s - s|}_{\text{share prediction error} \times \text{physics amplification}}
+
\underbrace{\text{remainder}}_{\text{residual}}
\]

其中 \(L_T = |C_{\text{renew}} - C_{\text{nonrenew}}|\) 是每个区域固定的物理放大系数。

这意味着：**Share prediction 误差在不同区域的传播系数不同**。标准 ERM 最小化所有区域的 CIF error 之和，会导致模型过度关注高 \(L_T\) 区域（物理放大系数大），而忽视低 \(L_T\) 区域的 share 预测质量。

Phys-IRM 的核心是：**学习一个 share predictor，使得它在所有区域上对 share 本身的预测风险都处于局部最优**。这等价于学习对 \(L_T\) 不变（即不受物理放大系数影响）的表征。

### 3.2 技术方案

#### 3.2.1 物理不变风险定义

标准 IRM 目标是：
\[
\min_{\phi, w} \sum_{e\in\mathcal{E}} \mathcal{R}^e(w\circ\phi)
\quad \text{s.t.} \quad w \in \arg\min_w \mathcal{R}^e(w\circ\phi), \forall e
\]

Phys-IRM 将风险重新定义为 **share-level risk**（而非 CIF-level risk）：

\[
\mathcal{R}^e_{\text{phys}}(\phi, w) = \mathbb{E}_{(x,y)\sim e}\left[\frac{1}{L_T^e} \cdot \ell(w\circ\phi(x), s_{\text{true}})\right]
\]

即对每个区域的 share prediction loss 用 \(1/L_T\) 做重要性加权，抵消物理放大系数的差异。

#### 3.2.2 训练算法

```
算法：Phys-IRM Training

输入: 源区域集合 S, 每个区域的 L_T^e

for each epoch:
    for each source region e in S:
        # Step 1: 用 1/L_T^e 加权计算 share loss
        L_share = mean( (1/L_T^e) * (ŝ - s_true)^2 )
        
        # Step 2: 物理层将 share 转为 CIF
        CIF_pred = ŝ * ef_r + (1-ŝ) * ef_nr
        
        # Step 3: CIF 监督（不加权，保证 CIF 质量）
        L_cif = mean( (CIF_pred - CIF_true)^2 )
        
        # Step 4: IRM 惩罚项：share predictor 在各区域都是最优的
        L_irm = ||∇_w L_share||_2^2
        
        # 总损失
        L_total = L_share + λ_cif * L_cif + γ * L_irm
```

#### 3.2.3 理论贡献

可以证明（基于 Theorem 1 的扩展）：

**Theorem 3（Phys-IRM 的泛化保证）**：
设 \(h\) 为 share predictor，若 \(h\) 在所有源区域的 share-level risk 上都达到 \(\epsilon\)-局部最优，则对任意满足配置连续性条件的目标区域 \(T\)，其 CIF error 满足：

\[
\mathcal{R}^T_{\text{CIF}}(h) \leq L_T \cdot \epsilon + \mathcal{O}(\text{dist}(c_T, c_{\text{nearest}}))
\]

其中 dist 是配置空间中的距离。

证明思路：利用 Theorem 1 的误差分解和 IRM 的不变性保证，结合配置连续性的 Lipschitz 条件。

### 3.3 创新空间

| 维度 | 评估 |
|---|---|
| **方法新颖度** | ★★★★☆ IRM 在时序中已有探索，但与物理约束结合是首次 |
| **技术深度** | ★★★★★ 需要对 IRM 框架做非平凡扩展，且有理论保证 |
| **与现有工作区分** | ★★★★☆ 已有 IRM + 时序的工作，但物理域是全新的 |
| **实验充分性** | ★★★★☆ 需要对比标准 ERM、IRM、Phys-IRM |
| **目标会议** | NeurIPS / ICML |
| **审稿风险** | 需要清晰说明为什么 \(L_T\) 可以作为域标签 |
| **发表可行性** | **高**（理论 + 物理 + 实验三重优势，适合 NeurIPS） |

### 3.4 关键文献

- Arjovsky et al., Invariant Risk Minimization, 2019（IRM 原论文）
- Krueger et al., Out-of-Distribution Generalization via Risk Extrapolation (REx), ICML 2021
- Liu et al., Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning, ICML 2024
- AdaRNN: Adaptive Learning and Forecasting of Time Series, NeurIPS 2022
- RevIN: Reversible Instance Normalization, ICLR 2022

---

## 四、方向三：反事实域解耦与因果自适应

### 4.1 核心思想

**不同区域的 RenewShare 动态由不同的因果结构驱动**（太阳能主导、风电主导、混合、调度密集等）。将这些因果结构从观测数据中学习并解耦，实现跨因果结构的零样本迁移。

核心洞见：Theorem 2 揭示的 U 型难度曲线暗示存在潜在的因果因子。中等可再生占比的区域可能共享相似的因果结构（风光互补达到稳态），而低端和高端的因果结构截然不同。

### 4.2 技术方案

#### 4.2.1 因果解耦框架

```
观测序列 x (RenewShare)
     ↓
  Domain Encoder E_d
     ↓
  Domain-specific factor z_d (受 config c 调制的因果结构)
     ↓
  Causal Decoder D
     ↓
  Counterfactual prediction:
    "如果这个区域有另一个 config c'，其 RenewShare 会如何变化？"
```

具体实现：

1. **Domain encoder**：将 (x, c) 映射到域特定表示 z_d；
2. **Causal graph discovery**：在 z_d 的分布上运行 DyCAST 或类似的动态因果发现；
3. **Causal decoder**：以 z_d 和反事实配置 c' 为条件生成预测；
4. **物理层**：将反事实 share 预测映射为 CIF。

#### 4.2.2 反事实数据增强

利用物理层的可逆性生成反事实训练样本：

1. 取源区域 A 的 share 序列 \(s_A(t)\)；
2. 用物理层计算其 CIF：\(\text{CIF}_A(t) = s_A(t)·e_{r,A} + (1-s_A(t))·e_{nr,A}\)；
3. 反事实变换：用区域 B 的因子重新计算 CIF：
   \[
   \text{CIF}_B^{\text{counterfactual}}(t) = s_A(t)·e_{r,B} + (1-s_A(t))·e_{nr,B}
   \]
4. 将 (s_A, CIF_B) 作为区域 B 的反事实训练样本。

这为因果解耦提供了丰富的数据支撑。

#### 4.2.3 因果结构聚类

在源区域上：
1. 为每个区域学习动态因果图（变量包括 RenewShare 的滞后值、负荷、温度等）；
2. 对因果图进行聚类，发现 causal regimes；
3. 新目标区域只需判断属于哪个 causal regime，即可获得对应的预测器。

### 4.3 创新空间

| 维度 | 评估 |
|---|---|
| **方法新颖度** | ★★★★★ 因果解耦 + 时序域泛化是非常前沿的方向 |
| **技术深度** | ★★★★★ 需要因果发现、反事实推理、域解耦多方面技术 |
| **与现有工作区分** | ★★★★★ 物理层提供了天然的因果结构约束 |
| **实验充分性** | ★★★☆☆ 因果评估本身比较困难 |
| **目标会议** | ICML / NeurIPS（偏理论） |
| **审稿风险** | 因果发现的不确定性可能被质疑；实验验证较难 |
| **发表可行性** | **中高**（概念极强，但实现和评估的难度也高） |

### 4.4 简化版路线（降低风险）

如果不追求完整的因果 pipeline，可以做一个更可控的版本：

**Config-conditioned Domain Disentanglement**：
- 使用 VAE 风格的域编码器，将 share 序列分解为 domain-invariant（可跨域迁移的时序模式）+ domain-specific（由配置决定的偏置）；
- 在 domain-invariant 空间上进行 LORO 训练；
- 这本质上是一个更温和的 causal interpretation。

这种简化版仍然有创新性，且更容易实现和验证。

### 4.5 关键文献

- DyCAST: Learning Dynamic Causal Structure from Time Series, ICLR 2025
- Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning, ICML 2024
- Causal Representation Learning for Time Series: A Survey, arxiv
- CaRGI: Causal Representation Learning via Generative Intervention, 2025

---

## 五、方向四：上下文学习器用于时序预测

### 5.1 核心思想

**将 LLM 的 In-Context Learning（ICL）范式迁移到时序预测，使模型能通过观察少量"示例对"来适应新区域。**

注意：这不是像 Time-LLM 那样使用 LLM 做时序预测。而是借鉴 ICL 的核心机制——**通过 attention 在上下文窗口中利用示例进行类比推理**——来设计时序预测器。

### 5.2 技术方案

#### 5.2.1 上下文构建

对于目标区域 T：

```
Context Window = [
    (x_source_1, y_source_1),  # 示例 1: 源区域的一个 (输入, 输出) 对
    (x_source_2, y_source_2),  # 示例 2
    ...
    (x_source_m, y_source_m),  # 示例 m
    (x_target, ?)              # 查询：目标区域的输入，输出未知
]
```

示例的选择基于配置距离和动态相似度（与方向一的检索策略相同）。

#### 5.2.2 ICL 预测器架构

```
Context Window (m+1 pairs)
     ↓
  Causal Transformer（单向注意力，防止信息泄漏）
     ↓
  Last token prediction（类似 GPT 的 next-token prediction）
     ↓
  The output for x_target is generated
     ↓
  Physics Layer
     ↓
  CIF
```

关键设计：
- **单向因果注意力**：确保预测只依赖于之前的 token（包括示例和自身的输入），不能"看到"自己的输出；
- **位置编码**：区分"示例输入"、"示例输出"、"查询输入"三种角色；
- **训练目标**：在源区域上训练，每个训练样本构造为 (m个示例, 1个查询) 的格式。

#### 5.2.3 零样本推理

给定新区域 T：
1. 用配置距离从源区域中选取 m 个最近区域各取一个典型窗口作为示例；
2. 将示例 + 目标输入拼接为 context window；
3. 一次前向传播获得预测。

这与 In-context Time Series Predictor (ICLR 2025) 的核心区别：
- ICTSP 使用交叉注意力在时序维度上进行上下文学习；
- 本方案使用因果 Transformer 实现更直接的 ICL 范式；
- 本方案的物理层确保输出的 CIF 物理一致性。

#### 5.2.4 训练细节

- Pre-training：在所有源区域的窗口对上训练（自监督形式）；
- Fine-tuning：在 LORO 设置下微调 ICL 能力；
- 对比学习辅助：正例是配置最近区域的窗口，负例是最远区域的窗口。

### 5.3 创新空间

| 维度 | 评估 |
|---|---|
| **方法新颖度** | ★★★★☆ ICL 在时序中研究刚起步 |
| **技术深度** | ★★★★☆ 需要设计 context 构建、attention 机制、训练策略 |
| **与现有工作区分** | ★★★★★ ICTSP (ICLR 2025) 是最近的，但使用的是交叉注意力而非 causal ICL |
| **实验充分性** | ★★★★☆ 需要大量消融（示例数量、选择策略等） |
| **目标会议** | ICLR / NeurIPS |
| **审稿风险** | "What's different from ICTSP?" — 必须有清晰区分 |
| **发表可行性** | **高**（ICL 概念有吸引力，物理层提供独特约束） |

### 5.4 关键文献

- In-context Time Series Predictor, ICLR 2025
- Brown et al., Language Models are Few-Shot Learners (GPT-3), NeurIPS 2020
- AutoTimes: Autoregressive Time Series Forecaster, NeurIPS 2024
- Timer-XL: Long-context Transformers for Unified Time Series Forecasting, ICLR 2025

---

## 六、方向五：去偏一致性分层预测

### 6.1 核心思想

**通过同时预测多个时间粒度（小时、日、周），并在物理层添加跨粒度一致性约束，实现自监督的去偏训练。**

核心观察：CIF 的日平均可以由小时值精确计算，且两者之间必须满足物理一致性。如果模型预测的 24 小时 CIF 的平均值与直接预测的日 CIF 不一致，这种不一致本身就是训练信号。

### 6.2 技术方案

#### 6.2.1 多粒度预测

```
输入: 336h RenewShare
     ↓
  共享 Encoder
     ↓
  ┌───────────┬───────────┬───────────┐
  │  Hourly   │  Daily    │  Weekly   │
  │  Head     │  Head     │  Head     │
  │  ŝ_1..ŝ_24│  s̄_day    │  s̄_week   │
  └───────────┴───────────┴───────────┘
     ↓
  Physics Layer（每个粒度独立应用）
     ↓
  Consistency Constraint:
    L_consist = ||mean(CIF_hourly) - CIF_daily||
```

#### 6.2.2 物理一致性损失

完整的损失函数：

\[
\mathcal{L} = \mathcal{L}_{\text{hourly}} + \mathcal{L}_{\text{daily}} + \mathcal{L}_{\text{weekly}} + \lambda \cdot \mathcal{L}_{\text{consist}}
\]

其中：
- \(\mathcal{L}_{\text{consist}} = \|\frac{1}{24}\sum_{h=1}^{24} \widehat{\text{CIF}}_h - \widehat{\text{CIF}}_{\text{day}}\|^2\)
- 以及：\(\mathcal{L}_{\text{consist\_week}} = \|\frac{1}{7}\sum_{d=1}^{7} \widehat{\text{CIF}}_{\text{day},d} - \widehat{\text{CIF}}_{\text{week}}\|^2\)

#### 6.2.3 去偏机制

一致性约束的作用类似于去偏正则化：
- 小时预测可能对特定小时的噪声敏感；
- 日预测更鲁棒但丢失细粒度信息；
- 一致性约束迫使两者在物理上一致，从而去除系统性偏差。

如果某个小时的预测系统性偏高，日平均会不匹配，梯度回传纠正小时预测器。

#### 6.2.4 与现有工作的区分

标准的分层时序预测（Hierarchical Time Series Forecasting）关注的是"实体层级"（国家→州→城市），一致性通过数学调和实现。本方案的不同之处在于：
1. 分层维度是**时间粒度**而非实体层级；
2. 一致性约束来自**物理层**（CIF 重建），而非人工定义的和；
3. 一致性作为**自监督训练信号**，而非后处理修正。

### 6.3 创新空间

| 维度 | 评估 |
|---|---|
| **方法新颖度** | ★★★☆☆ 分层预测本身成熟，但时间粒度 + 物理一致性是新的 |
| **技术深度** | ★★★☆☆ 实现相对直接 |
| **与现有工作区分** | ★★★☆☆ 需要明确与经典 reconciliation 的区别 |
| **实验充分性** | ★★★★☆ 可以单独验证一致性消除偏差 |
| **目标会议** | KDD / AAAI |
| **审稿风险** | "Isn't this just multi-task learning?" — 需强调一致性约束 |
| **发表可行性** | **中**（适合做补充实验或 workshops，单独投稿需更强的 motivation） |

### 6.4 关键文献

- Learning Optimal Projection for Forecast Reconciliation of Hierarchical Time Series, ICML 2024
- RHiOTS: A Framework for Evaluating Hierarchical Time Series Forecasting Algorithms, KDD 2024
- Hyndman et al., Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series, 2011

---

## 七、方向总览与投稿策略

### 7.1 五方向对比矩阵

| 方向 | 创新度 | 实现难度 | 理论深度 | 预期增益 | 目标会议 | 发表可行性 |
|---|---|---|---|---|---|---|
| RAG-TS | ★★★★★ | ★★★☆☆ | ★★★☆☆ | 中高 | NeurIPS/ICML | **高** |
| Phys-IRM | ★★★★☆ | ★★★★☆ | ★★★★★ | 中 | NeurIPS/ICML | **高** |
| Causal-ZS | ★★★★★ | ★★★★★ | ★★★★★ | 高（如果做成） | ICML/NeurIPS | 中高 |
| IC-TSF | ★★★★☆ | ★★★★☆ | ★★★☆☆ | 中高 | ICLR/NeurIPS | **高** |
| Debiased-Hier | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | 低中 | KDD/AAAI | 中 |

### 7.2 推荐投稿路径

#### 路径 A：主攻 NeurIPS 2027（2027 年 5 月 deadline）

**推荐主方向：RAG-TS + Phys-IRM 双投稿**

具体策略：
- **主论文**：RAG-TS — 概念新颖、故事完整、实验丰富
- **备选/同时投稿**：Phys-IRM — 理论扎实，适合 NeurIPS 的偏好

#### 路径 B：主攻 ICLR 2027（2026 年 10 月 deadline）

**推荐主方向：IC-TSF (In-context Learning)**

理由：ICLR 对 ICL 概念有天然偏好（ICLR 2025 已有 ICTSP），且审稿人对新兴概念更包容。时间较紧，但如果 IC-TSF 实现简单且效果显著，可冲刺。

#### 路径 C：KDD 2027（2027 年 2 月 deadline）

**推荐方向：RAG-TS + 应用 emphatic story**

理由：KDD 偏好"方法新颖 + 应用 solid"的论文，碳强度预测本身就是一个好的应用场景。时间充裕，适合作为备选。

### 7.3 联合投稿策略（最大化命中率）

| 阶段 | 时间 | 动作 |
|---|---|---|
| Phase 1 | 2026.08-09 | RAG-TS 核心实现 + 初步实验 |
| Phase 2 | 2026.09-10 | 如结果好 → 投 ICLR 2027（RAG-TS 或 IC-TSF） |
| Phase 3 | 2026.10-2027.01 | Phys-IRM 实现 + 理论证明 + RAG-TS 加强版 |
| Phase 4 | 2027.01-02 | 如 ICLR 未中 → 改写投 KDD 2027 或 ICML 2027 |
| Phase 5 | 2027.02-05 | Phys-IRM 完善 → 投 NeurIPS 2027 |
| Phase 6 | 2027.05+ | 根据反馈迭代其他方向 |

### 7.4 同时进行但不作为主攻的方向

- **Debiased-Hier**：可以作为 RAG-TS 或 Phys-IRM 的附加实验，单独成文竞争力不足
- **Causal-ZS 简化版**：Config-conditioned Domain Disentanglement 可以作为 Phys-IRM 的增强
- **MoE**：如果 RAG-TS 的检索机制可以替换为 learnable routing，可以作为对比或补充

---

## 八、短期行动计划

### 8.1 本周（8 月第一周，2026.08.04-08.10）

1. ~~完成代码清理的最终验证（所有脚本跑通一遍）~~ ✅ 已完成
2. 与导师讨论五个方向的优先级
3. 开始阅读 RAG-TS 和 Phys-IRM 相关文献，撰写文献笔记
4. 搭建 RAG-TS 的实验框架骨架代码

### 8.2 两周内（8月中旬）

1. 实现 RAG-TS 的核心检索模块：
   - Memory bank 构建
   - 配置距离 + DTW 相似度
   - top-k 检索
2. 在现有 `AdaptivePersistDLinear` 上接入检索增强
3. 跑第一版 LORO 对比实验（目标：初步验证检索增强是否有效）
4. 开始撰写 Phys-IRM 的理论推导

### 8.3 一个月内（9月初）

1. RAG-TS 全量实验完成（29 区域 × 多种检索策略）
2. 与当前 baseline 对比分析
3. Phys-IRM 初步实验验证
4. 决定主攻方向并开始论文写作
5. 如果冲刺 ICLR 2027（deadline ≈ 2026.10），此时应完成 outline 和核心实验

### 8.4 里程碑节点

| 日期 | 里程碑 |
|---|---|
| 2026.09.01 | RAG-TS v0 实验完成 |
| 2026.09.15 | 确定主攻方向，paper outline 完成 |
| 2026.10.01 | 如果 RAG-TS 或 IC-TSF 结果足够强，提交 ICLR 2027 |
| 2026.11.30 | 核心实验矩阵完成 |
| 2027.01.15 | 论文初稿完成（ICML 2027 或 KDD 2027） |
| 2027.02.01 | KDD 2027 / ICML 2027 投稿 |
| 2027.05.01 | NeurIPS 2027 投稿（最强版本） |

---

## 附录：方向交叉融合的可能性

### A.1 RAG-TS + Phys-IRM

检索到的样本可以作为 Phys-IRM 中的"虚拟域"，扩展域泛化的训练数据。每次检索相当于动态构建一个新域，IRM 在这些域上追求不变性。

### A.2 RAG-TS + IC-TSF

检索结果直接作为 ICL 的示例对(示例输入, 示例输出)，形成"检索增强上下文学习"（RA-ICL）—— 这是我认为最有潜力的融合方向。

### A.3 IC-TSF + Phys-IRM

ICL 的示例选择不仅仅是配置距离，还可以加入 IRM 的不变性约束——选择那些"如果不满足不变性条件就会导致高 regret"的示例来增强泛化。

### A.4 最终建议

**最推荐的路径**：以 **RAG-TS** 为主攻方向，引入 **RA-ICL** 作为核心机制（检索 + 上下文学习），用 **Phys-IRM** 的理论框架提供泛化保证。

这个组合具有：
- RAG 的概念吸引力（审稿人关注度）
- ICL 的技术前沿性（ICLR/NeurIPS 热点）
- 物理约束的独特优势（与所有 LLM-based 方法区分）
- 理论保证的深度（Theorem 3）

论文标题构想：
> **"Retrieval-Augmented In-Context Forecasting with Physics-Informed Invariant Learning"**
> 或更简洁：**"RA-ICF: Retrieval-Augmented In-Context Forecasting via Physics-Informed Invariant Learning"**

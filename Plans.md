# Plans — TransCIF 五方向融合模型 (Fused-5)

**Purpose:** 将 RAG / Phys-IRM / Causal / ICL / Hier 五个独立零样本方向融合为单一模型，目标 MAE < 41 gCO2/kWh（超过 PatchTST-supervised 41.465 基线），并为论文写入可信证据。

**Lane:** `[lane:gate]` — 涉及多 task / 多 file / product behavior / 论文主张，需要 stage gate（先修 bug → sanity → headline → ablation → full eval → paper）。

**团队验证模式:** `subagent` — 已并行 Architecture / Skeptic 视角；Code-review 视角命中 API 错误，其 findings 由 architect 与本地代码读取覆盖（line-level bugs 已列在 Task 1.2）。

**Spec skip reason:** 仓库无 root `spec.md`；产品契约由 `docs/paper/2026-07-26-zeroshot-config-cif-paper-zh.md` 持有。融合相关的契约变更通过更新论文稿（Phase 6）落地，不新建独立 spec 文件，避免与论文稿双源。

---

## 关键风险（来自 subagent 评审，必须显示在 plan 顶部）

| # | 风险 | 缓解 |
|---|------|------|
| R1 | **ZS+ 双计数陷阱**：在 cif 级 softmax 融合后再过 ZS+，等于两层 softmax 串联，可能抹掉 ZS+ 的逐日自适应 | 在 Task 3.3 同时报告 `Fused` 与 `Fused+ZS+` 两列，并强制做 `equal-weight-then-ZS+` 对照（Task 4.2）。如果后者 ≥ 前者，meta-learner 视为死重 |
| R2 | **Meta-overfit on ~28 sources**：region-level 方差主导 window-level，head 学到"哪个区域"而非"哪个方法" | 用 LOO CV 训练 head（Task 3.2），强制非负权重 + L2 + entropy floor，优先 softmax 而非 MLP 变体；报告各 fold 权重稳定性 |
| R3 | **多样性是幻觉**：Hier alone MAE 77.6 是坏的预测器，不是弱学习器；ZS+ 才是真正的工作马 | Drop-one ablation（Task 4.1）。若 "fusion − Hier" 不再击败 best-single，结论是"fusion 只是 ZS+ 的 wrapper"，论文必须重写 |
| R4 | **目标 <41 在噪声层内**：4-AU × seed-0 单点 <41 不可信；29-region 中位标准误约 σ/√29 ≈ 1–2 | Phase 5 强制 29-region × 5 seeds + 配对 Diebold-Mariano；论文 headline 只在统计显著时下结论 |
| R5 | **现有脚本三处 bug**（已确认）：`parameters()` 返回 list、`_i[0]` 计数器从不自增、FusionHead 无 L2/val split | Phase 1 整体重写为 `src/transcif/models/zeroshot/fusion.py`，旧脚本删除 |

---

## Task 表

> DoD 写在每行；`[tdd:required]` 表示实现前先写失败测试，`[tdd:skip:<reason>]` 表示省略。Lane/stage 写在内容首行。

### Phase 1 — 修复 & 重构（gate: 在此之前任何融合结果都不可信）

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 1.1 | `[lane:gate] [stage:plan]` 创建 `src/transcif/models/zeroshot/fusion.py`，定义 `FusionModel` 类与公开接口 `train_fusion(...) -> FusionModel`、`FusionModel.predict_cif(x_rs, config, ef_r, ef_nr) -> (n, HORIZON)`、`FusionModel.share_fn(x_window_np) -> (horizon,)`。`share_fn` 必须能直接喂给 `zs_plus_predict(..., share_fn=...)`（参见 `src/transcif/calibration/zs_plus.py:33`）。`[tdd:required]` | 接口签名单元测试通过：构造 dummy 5-stack → `predict_cif` 输出形状 `(n, HORIZON)`；`share_fn` 输出形状 `(HORIZON,)` 且值落在 [0, 1] | - | `cc:完了` |
| 1.2 | `[lane:gate] [stage:tdd]` 修复 `run_fused_five.py` 三处 bug 并迁移到新模块：(a) `_FuseModel.parameters()` 返回 list 不是 iterator → `next()` 崩；(b) `_FuseModel.__call__` 用 `self._i[0]` 但从不自增 → 每个 origin 都返回 row 0；(c) `FusionHead` 训练无 val split / 无 L2 / 无 early stop。`[tdd:required]` | 删除 `_FuseModel` 与 `FusedModel` 占位类；`run_fused_five.py` 仅保留 argparse + 数据加载 + 调用 `train_fusion` + 调用 `zs_plus_predict`；脚本主体 < 80 行 | 1.1 | `cc:完了 [f91bbbc]` |
| 1.3 | `[lane:gate] [stage:tdd]` 实现 zero-shot 训练数据收集器：对每个 source region 跑 5 个方向，组装 `(n_i, 5, HORIZON)` cif stack 与 `(n_i, HORIZON)` true cif；显式只取 source 的 TEST 窗口（避免与 source 的训练窗泄漏）。`[tdd:required]` | 单元测试：mock 5 个方向 → 收集器返回的 stack shape 与 true cif shape 一致；source 训练窗的索引不与 source TEST 窗口的起点重叠 | 1.1 | `cc:完了 [f91bbbc]` |
| 1.4 | `[lane:gate] [stage:review]` Phase 1 smoke：`.venv/bin/python scripts/experiments/run_fused_five.py --regions AU1 AU2 --seed 0` 必须完整跑完不报错，写出非空 JSON（至少 2 行） | `results/fused_five_smoke.json` 含 2 个 region 的 `transcif_fused5` + `transcif_fused5_plus` 字段，MAE 是有限正数 | 1.2, 1.3 | `cc:完了 [5f5f7bf]` |

### Phase 2 — Baseline 融合（gate: 确认管道活着）

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 2.1 | `[lane:gate]` 在 `fusion.py` 增加 `EqualWeightFusion` 与 `MedianFusion` 两个无参 baseline | `compute_metrics` 输出有限；equal-weight 与 median 是两条独立 code path | 1.1 | `cc:完了 [9dacec1]` |
| 2.2 | `[lane:gate] [stage:review]` 4-AU × seed-0 sanity：跑 equal-weight / median / softmax-head（fixed）三组，确认 (a) 都不崩；(b) equal-weight 不显著差于 softmax-head（如果显著差，说明 head 有信号；如果持平，按 skeptic R2 的预判 head 是死重） | `results/fused_five_sanity.json` 含 3 个变体 × 4 region 的 MAE/RMSE/sMAPE；写入一行 markdown 总结到 `results/fused_five_sanity.md` | 2.1, 1.4 | `cc:完了` |

### Phase 3 — Headline 融合：非负基底混合（论文贡献）

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 3.1 | `[lane:gate] [stage:tdd]` 实现 `BasisMixFusion`：5 → 1 softmax 权重 + (a) 非负约束（softmax 天然满足）+ (b) L2 权重正则 + (c) entropy floor（强制权重不塌缩到单点）+ (d) diversity reg（pairwise weight-cosine 惩罚，鼓励 5 个方向都被使用）。论文 framing：每个方向 = 命名基底（knowledge/physics/causality/context/hierarchy）。`[tdd:required]` | 单元测试：极端同质 stack → entropy floor 阻止塌缩；loss 可微；权重和 = 1 | 1.1 | `cc:完了 [9dacec1]` |
| 3.2 | `[lane:gate] [stage:tdd]` LOO-CV 训练 pipeline：留一 source region 出，剩余 source region 训练 head，预测被留出 region 的 cif；遍历所有 source region，得到 OOF（out-of-fold）预测；最终 head 用全部 source 重训一次。报告各 fold 权重向量与 OOF MAE。`[tdd:required]` | OOF MAE 与 in-fold MAE 差距 < 20%（差距过大 = 过拟合标志）；5 fold 的权重向量 std < 0.15 | 3.1, 1.3 | `cc:完了 [5f5f7bf]` |
| 3.3 | `[lane:gate] [stage:review]` ZS+ 集成与 R1 双计数检查：同时输出 (a) `BasisMix_fused`（cif 级融合，不过 ZS+），(b) `BasisMix_fused_plus`（融合后过 ZS+），(c) `equal_weight_then_plus`（R1 控制组） | 三组指标都在 `results/fused_five_headline.json`；如果 (c) ≥ (b) 的中位 MAE，写入 `results/fused_five_headline.md` 标记 "meta-learner is dead weight, headline 退回到 equal-weight+ZS+" | 3.2, 2.1 | `cc:完了` |

### Phase 4 — Drop-one 消融（gate for R3: 证明不是 ZS+ wrapper）

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 4.1 | `[lane:gate]` Drop-one-direction pipeline：5 次重训 BasisMixFusion，每次去掉一个方向；产出 5 组 `(drop_which, MAE, RMSE, sMAPE)`；附上 "fusion using only 4 of 5" 的 ZS+ 与 non-ZS+ 双列 | `results/fused_five_dropone.json`；附 markdown 表 `results/fused_five_dropone.md` 列出 "drop X → MAE Y" | 3.2 | `cc:完了` |
| 4.2 | `[lane:gate] [stage:review]` 关键判定：drop Hier（最弱方向）后 fusion 是否仍击败 best-single（Causal 41.18 中位 / 42.10 单点）？若是 → 多样性真实；若否 → R3 命中，论文重定位为 "ZS+ wrapper"。把判定写入 `results/fused_five_dropone.md` 顶部 verdict 行 | verdict 行必须明确写 `DIVERSITY_REAL` 或 `ZS_PLUS_WRAPPER`，并附 drop-Hier MAE 数字 | 4.1 | `cc:完了` |

### Phase 5 — 全规模评测（gate for paper headline claim）

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 5.1 | `[lane:gate]` 29-region LORO × 5 seeds 全量评测：对每个 target region 跑（每个方向单跑 + 等权融合 + BasisMix + BasisMix+ZS+），统一写入 `results/fused_five_full.json`。预算估计 ~ 29 region × 5 seed × ~6 分钟 ≈ 14.5 小时；可拆 5 个 seed 进程并行 | JSON 含 29 × 5 = 145 行，每行所有方法的 mae/rmse/smape；`results/fused_five_full_summary.json` 含中位与 mean ± std | 3.3, 4.2 | `cc:TODO` |
| 5.2 | `[lane:gate]` 统计显著性：per-region paired Diebold-Mariano 检验（BasisMix+ZS+ vs PatchTST-supervised、BasisMix+ZS+ vs Causal-alone、BasisMix+ZS+ vs equal-weight+ZS+）；输出 p-value 矩阵与 Holm-Bonferroni 校正后的显著性 | `results/fused_five_significance.json` 含 p-value、n_regions where significant at α=0.05；markdown 表格 `results/fused_five_significance.md` | 5.1 | `cc:TODO` |
| 5.3 | `[lane:gate] [stage:review]` Paper-claim verdict：基于 5.1/5.2 写一行结论 — (a) BasisMix+ZS+ 中位 MAE < 41 且 DM 检验显著 → headline 成立；(b) 中位 < 41 但不显著 → 降级为 "competitive with supervised"；(c) 中位 ≥ 41 → headline 失败，论文回到 "5 单方向独立 + ZS+" 故事 | `results/fused_five_verdict.md` 一页结论 + 三种情况各自对应的论文措辞建议 | 5.2 | `cc:TODO` |

### Phase 6 — 论文写入

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 6.1 | 在 `docs/paper/2026-07-26-zeroshot-config-cif-paper-zh.md` 与英文版新增 "Section X: Five-Prior Basis Fusion"，包含：动机（5 个方向 = 5 个互补先验）、方法（非负基底混合 + LOO-CV + ZS+ 集成）、表（headline 数字 + drop-one + 显著性）、限制（R3 verdict）。英文版同步 | 两份论文稿都新增完整章节；数字与 `results/fused_five_full_summary.json` 一致；reviewer-readable | 5.3 | `cc:TODO` |
| 6.2 | 生成图：(a) 5 方向权重分布箱线图（5 seeds × 29 regions）；(b) drop-one MAE 条形图；(c) per-region BasisMix+ZS+ vs PatchTST 散点（45° 线 + 1:1.25 线） | `figures/fusion_weights.png`、`figures/fusion_dropone.png`、`figures/fusion_per_region.png` 三个文件，300 DPI | 5.3 | `cc:TODO` |

### Phase 7 — 可复现性

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 7.1 | `[lane:release]` 更新 `README.md` 复现段：列出 `run_fused_five.py --regions ... --seed ...` 与 `run_fused_five_full.sh`（新建）的命令，预计时长 | README 复现段含融合部分；`scripts/experiments/run_fused_five_full.sh` 可一键起 5 seed | 5.1 | `cc:TODO` |
| 7.2 | `[lane:release]` CI smoke：在 `tests/` 加 1 个集成测试 `tests/test_fusion_smoke.py`，跑 1 region × 1 seed × 最少 epoch，确保未来 PR 不会再次破坏 `parameters()` 这类接口 | `pytest tests/test_fusion_smoke.py` 在 < 60 秒内通过 | 1.4 | `cc:TODO` |

---

## 事前确认

- 事項: 长时间训练运行 (Phase 5.1 全量评测预计 14.5 小时)
  理由: 29-region × 5-seed LORO 评估是论文 claim 的最低可信证据；CPU 跑不动必须 GPU
  scope: Phase 5 / Task 5.1
- 事項: 大量 results JSON 与 figure 写入 (`results/fused_five_*.json`, `figures/fusion_*.png`)
  理由: 评测产物落盘供论文与 README 引用
  scope: Phase 3-7 / Task 3.3, 4.1, 5.1, 6.2
- 事項: 论文稿双源同步编辑 (`docs/paper/2026-07-26-zeroshot-config-cif-paper-zh.md` 与 `-en` 版)
  理由: Task 6.1 双语版必须数字一致；如不同步会有论文双源漂移
  scope: Phase 6 / Task 6.1

无 secret-read、无外部发送、无破坏性操作。Phase 5.1 长时间运行建议用 `run_in_background` 或独立 shell；不阻塞会话。

---

## 完成后启动指引

- 新会话启动命令: `claude`
- 启动后第一个输入: `/harness-work 1.1`
- 适合的场景: Phase 1 的四个 task 有强依赖（1.2 / 1.3 / 1.4 都依赖 1.1），不适合并行；按 task 编号顺序推进，单 task 单 session 最稳

进入 Phase 2 之后切换:
- 启动后第一个输入: `/breezing all`
- 适合的场景: Phase 2 的两个 task 与 Phase 3.1 互相独立，可并行

进入 Phase 5 长时间评测:
- 新会话启动命令: `ENABLE_PROMPT_CACHING_1H=1 claude`
- 启动后第一个输入: `/harness-loop 5.1`
- 适合的场景: 29-region × 5-seed 评测预计 14+ 小时，需要长 run + resume 能力

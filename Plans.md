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
| 5.1 | `[lane:gate]` 29-region LORO × 5 seeds 全量评测：对每个 target region 跑（每个方向单跑 + 等权融合 + BasisMix + BasisMix+ZS+），统一写入 `results/fused_five_full.json`。预算估计 ~ 29 region × 5 seed × ~6 分钟 ≈ 14.5 小时；可拆 5 个 seed 进程并行 | JSON 含 29 × 5 = 145 行，每行所有方法的 mae/rmse/smape；`results/fused_five_full_summary.json` 含中位与 mean ± std | 3.3, 4.2 | `cc:完了` |
| 5.2 | `[lane:gate]` 统计显著性：per-region paired Diebold-Mariano 检验（BasisMix+ZS+ vs PatchTST-supervised、BasisMix+ZS+ vs Causal-alone、BasisMix+ZS+ vs equal-weight+ZS+）；输出 p-value 矩阵与 Holm-Bonferroni 校正后的显著性 | `results/fused_five_significance.json` 含 p-value、n_regions where significant at α=0.05；markdown 表格 `results/fused_five_significance.md` | 5.1 | `cc:完了` |
| 5.3 | `[lane:gate] [stage:review]` Paper-claim verdict：基于 5.1/5.2 写一行结论 — (a) BasisMix+ZS+ 中位 MAE < 41 且 DM 检验显著 → headline 成立；(b) 中位 < 41 但不显著 → 降级为 "competitive with supervised"；(c) 中位 ≥ 41 → headline 失败，论文回到 "5 单方向独立 + ZS+" 故事 | `results/fused_five_verdict.md` 一页结论 + 三种情况各自对应的论文措辞建议 | 5.2 | `cc:完了` |

### Phase 6 — 论文写入

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 6.1 | 在 `docs/paper/2026-07-26-zeroshot-config-cif-paper-zh.md` 与英文版新增 "Section X: Five-Prior Basis Fusion"，包含：动机（5 个方向 = 5 个互补先验）、方法（非负基底混合 + LOO-CV + ZS+ 集成）、表（headline 数字 + drop-one + 显著性）、限制（R3 verdict）。英文版同步 | 两份论文稿都新增完整章节；数字与 `results/fused_five_full_summary.json` 一致；reviewer-readable | 5.3 | `cc:TODO` |
| 6.2 | 生成图：(a) 5 方向权重分布箱线图（5 seeds × 29 regions）；(b) drop-one MAE 条形图；(c) per-region BasisMix+ZS+ vs PatchTST 散点（45° 线 + 1:1.25 线） | `figures/fusion_weights.png`、`figures/fusion_dropone.png`、`figures/fusion_per_region.png` 三个文件，300 DPI | 5.3 | `cc:TODO` |

### Phase 7 — 可复现性

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 7.1 | `[lane:release]` 更新 `README.md` 复现段：列出 `run_fused_five.py --regions ... --seed ...` 与 `run_fused_five_full.sh`（新建）的命令，预计时长 | README 复现段含融合部分；`scripts/experiments/run_fused_five_full.sh` 可一键起 5 seed | 5.1 | `cc:完了`（README 加 Phase 8/9 joint-train 复现段 + headline 行；native pipeline 命令含 `--gate`） |
| 7.2 | `[lane:release]` CI smoke：在 `tests/` 加 1 个集成测试，跑 1 region × 1 seed × 最少 epoch，确保未来 PR 不会再次破坏 `parameters()` 这类接口 | 集成测试 < 60 秒通过 | 1.4 | `cc:完了`（`tests/test_native_pipeline_smoke.py` 4 测试 < 2s：覆盖 native_stage / eval_held_out / assemble_stack / gate snapshot-restore / head_modules 全 live 方向） |

### Phase 8 — 可微 ZS+ joint training rescue（条件性，12 GPU-hr 予算）

**Spec skip reason**: この Phase は `src/transcif/calibration/zs_plus.py` のアーキテクチャ変更（非微分 → 微分可能）と学習 pipeline の追加。product contract（零ショット CIF 予測）は不変。`docs/paper/...zeroshot-config-cif-paper-zh.md` への反映は Task 8.7 verdict が `case (a)`（< 41 達成）の場合のみ Phase 6 で実施。

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 8.1 | `[lane:gate] [stage:tdd]` 微分可能 ZS+ を `src/transcif/calibration/differentiable_zs_plus.py` に `DifferentiableZSPlus(nn.Module)` として実装。6 branch 全てを計算し、soft attention（過去 K 日の branch error を入力とした学習可能重み + branch gate）で融合。`[tdd:required]` | 単体テスト：(a) forward が `(n_origins, HORIZON)` を返す；(b) `share_fn` 入力に対して gradient が伝播する；(c) branches 1-4 が定義式（daily lag / weekly lag / 7-day mean / weekly mean）と一致；(d) QLD1 seed 0 + causal direction で MAE が [10, 80] の健全範囲（非微分 DEFAULT ~27 と同次元） | - | `cc:完了` |
| 8.2 | `[lane:gate] [stage:tdd]` Adversarial-persistence 損失を `src/transcif/training/adversarial_loss.py` に実装。`L = ReLU(pred_mae - persistence_mae * (1 - margin)).mean()`、`margin=0.10`。per-window 計算。`[tdd:required]` | 単体テスト：pred=persistence のとき L=mean(0.10 × persistence_mae)；pred=persistence×0.9 のとき L=0；gradient が active region で非ゼロ | 8.1 | `cc:完了` |
| 8.3 | `[lane:gate] [stage:tdd]` 3-stage warmup 学習 pipeline を `scripts/experiments/run_joint_train.py` に実装。Stage 1: 5 方向モデル凍結、ZS+ attention + BasisMix head のみ学習（~30 min）。Stage 2: 5 方向の出力層のみ解除（~60-90 min）。`[tdd:required]` | 単体テスト：1 source-target pair で end-to-end 完走 < 30 min；checkpoint 保存；stage 1 / stage 2 が別々の最適化ステップとして実行される | 8.2 | `cc:完了` |
| 8.4 | `[lane:gate] [stage:review]` 算力 sanity：QLD1 target × NSW1+VIC1+SA1 sources × seed 0 で学習。`results/joint_train_sanity.json` に QLD1 TEST windows の MAE を書き出し。 | JSON が QLD1 の `joint_trained` MAE を含む；学習時間 ≤ 2 GPU-hr；`results/joint_train_sanity.md` に loss 曲線と MAE サマリ | 8.3 | `cc:完了` |
| 8.5 | `[lane:gate] [stage:review]` Go/No-Go gate：(a) MAE < 41 → Phase 8.6へ。(b) 41 ≤ MAE < 46 → ハイパラ調整で再学習 1 回。(c) MAE ≥ 46 → Phase 8 中止、negative result 記録。判定は `results/joint_train_gate.md` に明示。 | gate markdown が (a)/(b)/(c) を明示；(b) の場合は再学習結果と再判定を含む；(c) の場合は中止理由と今後の推奨を含む | 8.4 | `cc:完了` |
| 8.6 | `[lane:gate]` 完全 LORO 評価：29 領域 × 5 seed で joint-trained モデルを評価。`results/joint_train_full.json` に 145 行。予算 ~7 GPU-hr。 | JSON が 145 行で `joint_trained` メソッドを含む；`results/joint_train_full_summary.json` が median/mean/std を含む；既存 `fused_five_full.json` と同じ schema で直接比較可能 | 8.5 | `cc:完了` |
| 8.7 | `[lane:gate] [stage:review]` 最終 verdict：joint-trained median MAE vs 41 目標、vs BasisMix+ baseline (46.89)。3 ケースを `results/joint_train_verdict.md` に書き出し：(a) < 41 → headline 復活；(b) 41-46 → competitive but not beating supervised；(c) ≥ 46 → joint training 効果なし、既存故事を維持 | verdict markdown が (a)/(b)/(c) を明示；数値根拠を含む | 8.6 | `cc:完了` |

### Phase 9 — Torch-native 方向模型 + 方案 A（并联可学习 fusion）

**决策依据**: Phase 8 verdict 指出 `(5,24)` correction 代理只捕获~30%信号，方向模型是 numpy/torch 混合体无法端到端反传。本 Phase 把 3 个"易迁移"方向（causal/phys/hier）做成可微，替换代理。用户原构思"5 方向串联"经论证否决（独立归纳偏置非 pipeline 阶段，误差累积），改取**方案 A：并联 + 可学习 fusion**。RAG/ICL 迁移成本高，先作冻结常量。

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 9.1 | `[lane:gate] [stage:tdd]` `src/transcif/models/zeroshot/native.py`：`TorchNativePredictor` ABC + `NativePhys/Causal/Hier` 薄包装（内联物理转换保梯度）+ `FrozenConstant`（RAG/ICL 断点）+ `LearnedFusion`（逐窗口权重）+ `pad_config_t`。零侵入，不改既有 `predict_*_zs` | 9 单测全绿：3 个 live wrapper 梯度可达模型参数；FrozenConstant 梯度止步；LearnedFusion 权重行和=1 且逐窗口变化；NativePhys 与手算数值一致 | - | `cc:完了` |
| 9.2 | `[lane:gate] [stage:tdd]` `scripts/experiments/run_joint_train_native.py`：2-stage pipeline。Stage1 方向全冻只训 fusion+ZS+；Stage2 解冻 3 方向**输出层**（DLinear heads / VAE predictor / hourly head）+fusion+ZS+。损失 MAE+0.5·adv-persist，weight_decay 1e-4 | 1 区域 end-to-end 完走；held-out MAE 写出；stage1/stage2 分别优化 | 9.1 | `cc:完了` |
| 9.3 | `[lane:gate] [stage:review]` 验证实验：QLD1（easy）+ UK_08（hard）× seed0，3 配置（baseline frozen-proxy / native-learned / native-softmax）。gate：native 比 baseline 低 ≥1.0 MAE 且不恶化另一区域 → 进 9.4 | `results/native_validation.json` + `results/native_validation_verdict.md`。实测 QLD1 Δ+1.81、UK_08 Δ+8.08，gate 通过；LearnedFusion≈softmax（fusion 升级中性，lift 来自 head unfreeze） | 9.2 | `cc:完了` |
| 9.4 | `[lane:gate]` 完全 LORO：29×5 seed，learned fusion，写 `results/joint_train_native_full.json`（145 行）+ `_summary.json`，与 `joint_train_full.json` 同 schema 直接对比 | 145 行；summary 含 median/mean/std；与 frozen-proxy baseline 40.53 对比 | 9.3 | `cc:完了` |
| 9.5 | `[lane:gate] [stage:review]` 最终对比 verdict：native median vs frozen-proxy 40.53、vs PatchTST 41.47。进 significance 框架（paired Wilcoxon + Holm） | `results/joint_train_native_verdict.md` + 显著性检验结果。实测 native median **39.53**（frozen 40.53，Δ+1.62，Wilcoxon p=5e-14，79% 胜；难区域 VIC1 +11.4、UK_08 +7.3；PatchTST 单样本 p=0.057 borderline） | 9.4 | `cc:完了` |
| 9.6 | `[lane:gate] [stage:tdd]` RAG `RagMemoryBank` 可微化（buffer 化 X/Y + matmul kNN + softmax 加权）与 ICL torch-native（per-query context 检索作 no_grad 预处理，transformer forward 保梯度，源窗口缓存）。`NativeRAG`/`NativeICL` 加入 `native.py`。至此 **5 方向全 torch-native**（3+2 frozen → 5 live + 0 frozen） | 单测：RAG/ICL 梯度可达模型参数；bank buffers 不进 optimizer；不破坏既有 zero-shot eval。实测 11 测试绿；QLD1 5-live 端到端跑通 | 9.1 | `cc:完了（代码）` |
| 9.7 | `[lane:gate]` 5-live + internal-val 门（eps=2.0，conservative：仅 Stage2 明显变差才回滚）全量 LORO。隔离 (a) 门对 3-live 的效果（gated-3live vs 39.53）、(b) RAG+ICL-native 的联合效果（5-live+gate vs 39.53） | `results/joint_train_native_5live_gated_full.json`（145 行）+ `joint_train_native_5live_verdict.md`。实测 median **39.04**；vs frozen-proxy +1.86（p=1e-14, 81% 胜）；vs PatchTST 单样本 **p=0.045**（首个跨过 α=0.05）；但 vs 3-live(39.53) 配对 p=0.12（噪声内，RAG/ICL-native+门无稳健额外增益）；门 9/145 回滚 | 9.6 | `cc:完了` |

### Phase FD — 燃料分解架构 TransCIF-FD(跨域冷启动:I_cfg 层级 + benchmark)

**决策依据**(2026-08-15,文献调研):EnsembleCI(arXiv 2505.01959)证明日历特征是跨网格最强特征、CarbonCast 的两层"先逐燃料发电再合成 CI"结构从未被用于零样本;HN-MVTS(AAAI 2026)证明元数据生成末层权重的超网络泛化优于 bias-only 条件化;中国官方仅有年度/省级平均因子,小时级"分时分区电碳因子"是前沿 → 新增 **I_cfg 层级**(仅 config+天气+日历,无遥测)作为主贡献与 benchmark 接口。仓库内 25/29 区域的逐小时逐燃料数据(`data_2023/fuel/`)从未进入 headline 模型。

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| FD-0 | 数据接口与物理特征:`config/region_meta.py`(29 区域 lat/lon/tz)、`physics/astro.py`(太阳高度角/晴空 GHI/风功率曲线)、`data/calendar.py`(本地时 sin/cos 日历)、`data/fuel.py`(逐燃料序列加载 + 5 通道天气 exog + `build_fd_windows` + `region_fuel_efs` 热力组 EF 重标定)+ `loaders.py` 增 `hours` 键 | 30 单测全绿;物理重建 CIF 中位 MAE ~8.9(25 燃料区域,South-East England 进口重组残差已知);主套件 129 passed 无回归 | - | `cc:完了` |
| FD-1a | `models/fuel_decomp.py`:FuelDecompNet(solar=天文包络×天气调制;wind=功率曲线归一;baseload=水平+支撑掩码;thermal=残差+config 对数锚定;聚合 rs 头带 logit 锚定与历史门;有界 EF 校正;~20k 参数;冷模式 dropout 双层级) | 9 单测全绿(simplex、冷模式不变性、天文相关性 >0.9、梯度、参数预算 <120k) | FD-0 | `cc:完了` |
| FD-1b | `models/zeroshot/fuel.py`:`train_fuel_zero_shot`(config 加权采样、EF 加权份额损失、p_cold/p_mix、共享绝对 origin 网格)+ `predict_fuel_windows` + shape 指标(diurnal/monthly-shape MAE、Spearman)+ `prepare_fd_region` | 单测 + 8 区域 smoke;关键 bug 修复:冷模式持久门、风速归一爆炸、热力拆分幻觉(config 锚定+支撑掩码)、logit 锚定跨模式纠缠 | FD-1a | `cc:完了` |
| FD-1c | `scripts/experiments/run_fuel_decomp_eval.py`:LORO 快速协议(8 区域×2 seeds×600 epochs,resume-safe),同时评 I_0 与 I_cfg + 基线(persistence/config-constant/monthly-constant oracle) | `results/fuel_decomp_eval_quick.json` 16 对;**I_0 中位 48.66 < 现有 ZS 52.1 ✓ gate**;I_cfg 73.30 < config-constant 84.79(中位)但配对 p=0.71 未过显著线;I_cfg Spearman 0.22 vs 常数 0(p=0.0017 ✓ 排序技能显著) | FD-1b | `cc:完了` |
| FD-1d | FD-1 verdict 文档 | `results/fuel_decomp_fd1_verdict.md`:**PARTIAL PASS**(I_0 48.66<52.1 ✓;I_cfg 排序技能显著 p=0.0017 ✓;水平 MAE 优于常数基线但配对不显著) | FD-1c | `cc:完了` |
| FD-2a | `models/hypernet.py`(ConfigHyperNet 生成 5 个动态头 111 权重,零初始化=FD-1 热启动)+ `training/synthetic.py`(GridRecombiner 物理引导网格重组 + neighborhood_batch)+ 训练器 `p_mix`/`use_hypernet` 接入 + ZS+ `share_fn` 钩子 | 8 单测全绿;混合 CIF 标签物理精确(重算而非线性混合);邻域 batch config 距离 < 源均值;ZS+ 管线集成冒烟通过 | FD-1b | `cc:完了` |
| FD-2b | 消融:p_mix=0.3 / hypernet / 二者并用,同 FD-1c 协议 | `results/fuel_decomp_eval_{mix,hn,mixhn}.json` + `fuel_decomp_fd2_verdict.md`:**NEGATIVE/MIXED**——hypernet I_cfg 中位 −8.8 但配对 p=0.98(VIC1/PJM 灾难漂移,SA1/UK_08 大赢);p_mix MAE 中性但冷偏差减半(−8.9→−3.5);组合负交互。FD-1 保持默认 | FD-2a | `cc:完了` |
| FD-3 | `docs/BENCHMARK.md`(任务/层级/合法性规则/指标/基线/排行榜 schema)+ `scripts/benchmark/run_benchmark.py`(汇总 leaderboard.json) | 文档 + 脚本可运行;`results/leaderboard.json` 含 legacy 阶梯(ZS 50.72 / ZS+ 46.80 / PatchTST 43.50,n=145)与 FD 双层级 | FD-1c | `cc:完了` |
| FD-5 | **TransCIF 经验移植**(FD-4 顺延):(a) ZS+ branch-0 水平锚定机制移植进模型 I_0 层(`anchor_gate`,门控,sigmoid 偏置 1.5 起步近锚定;观测 rs 流 → 2 燃料恒等式水平,I_0 合法不需 CIF 历史);(b) ZS+ 校准接入评估 I_+ 层(`make_zs_plus_share_fn` + `zs_plus_predict`,drop-one verdict 的主力机制) | **PASS**:I_+ 中位 **36.81** vs persistence 41.41,**16/16 全胜(p=3.1e-5)**,Spearman 0.412;I_0 48.66→45.13(UK_08 −23,p=0.10 次显著);I_cfg 无影响(p=0.94,冷模式关闭符合设计);单测 11 绿。`results/fuel_decomp_fd5_verdict.md` | FD-2b | `cc:完了` |
| FD-4 | 全量 LORO(29×5=145 对,与 legacy 同区域集同 seeds,含 UK_18_GB)+ donor/hypernet-stab 消融 | `results/fuel_decomp_fd4_verdict.md`:**equalizer 全尺度确认(异构骨干 r=0.9999,|Δ|中位 0.17)——I_+ 骨干无关,竞争在校准**;FD I_+ 47.08(vs persistence win 99%,p=1.6e-25;vs legacy ZS+ 46.80 平局);I_0 55.09 vs legacy 50.72(legacy 保持旗舰);**I_cfg 65.67 vs 年度常数 70.90,Spearman 0.280,77% 区域>0.1,该层级唯一方法**;donor 加权拒绝(p=0.025 反向);hypernet+0.15 界拒绝(VIC1 灾难未驯服) | FD-5 | `cc:完了` |
| FD-6 | I_cfg 形态提升:形态损失项(逐窗去均值 CIF MAE,`LAMBDA_SHAPE=0.5` 默认)——FD-4 证明水平近 oracle,形态是唯一空间;与锚定正交 | **PASS**:全量 145 对 I_cfg **Spearman 0.280→0.303(p=0.049,52% 区域>0.3)**,MAE 65.79 持平;I_+ 46.99(vs persistence win 99%,p=1.6e-25)。`fuel_decomp_fd6_verdict.md` | FD-4 | `cc:完了` |
| FD-7 | 中国省份月度接口:`build_monthly_config_table` + 逐窗口月度 config(1 月发布滞后)+ `--monthly-config` + `demo_cn_province.py` 部署演示 | **基准 NEGATIVE / 接口交付**:8×2 I_cfg 71.76→72.21(win 12%,p=0.01)——基准区域年结构平稳,月度分辨率=噪声;UK 可再生含核电定义使月度 mean_rs 失真。默认年度 config,月度保留选项。演示:虚拟华北煤主省份 7 月周预测 mean 497.6,**光伏峰→CIF 谷、煤电日内 0.68→0.54 让路**,物理正确。`fuel_decomp_fd7_verdict.md` | FD-6 | `cc:完了` |
| FD-8 | 日前协议合法性:`apply_day_ahead_weather_error`(未来天气退化至 NWP 技能,派生量一致重算)+ `--weather-noise` 配对轨道 + BENCHMARK.md 双轨道化 | **日前声明成立**:MAE 中位完全不变(45.68/71.76→45.68/71.71),均值 +1.7-1.8(n.s.);Spearman I_cfg 0.239→0.216(p=0.03);退化集中在风依赖网格(UK_08 +6.9)。信号主力=确定性通道(天文/日历/config)。`fuel_decomp_fd8_verdict.md` | FD-7 | `cc:完了` |
| FD-9 | I_J 联合微调(`run_fuel_joint.py`:12/12/3 origins 协议,冻结水平通路+解冻形态头+对抗 persistence+internal-val 门)+ MAE 目标对齐 | **NEGATIVE**:32.03→32.17(win 19%,p=0.016;门 5/16)——288h 标签改善校准而非骨干,ZS+ 已榨干(equalizer 第三变体);I_J 仍 100% 胜 persistence。**目标重述**:系统 median≤10 物理不可能(地板 50-53,天花板 30-35);分层现状 easy 6 区 I_+ 13.8(BPAT 6.2 已≤10)/medium 34.3/hard 57.0。`fuel_decomp_fd9_verdict.md` | FD-8 | `cc:完了` |
| FD-10 | 难度解剖(29 区域结构/波动/噪声 vs MAE 相关 + 效率比均匀性 + 标签噪声分解) | 结论:难度≈内在波动性(std r=+0.85/ramp r=+0.72),结构类型几乎无关(rs r=−0.04);效率比跨档均匀 0.93-0.96(难区是区域属性非模型失效);UK South-East 标签噪声占 10-37%(UK_14 vs 物理真值 44.0 vs 70.3);AU 无燃料遥测双重惩罚。方案:双轨标签/NEMED DUID/联络线通道/效率比主指标/概率预测。中国省份预期 easy-medium 档。`fuel_decomp_fd10_difficulty_anatomy.md` | FD-9 | `cc:完了` |
| FD-11 | 优化路线图(`docs/OPTIMIZATION_ROADMAP.md`):结构 5 档(A基荷/B风电/C光伏/D进口/E火电)× 数据条件;逐档专属方案(特征/损失/头);确定性结构路由(RegimeMoE 机制,零参数,避 hypernet 教训);中国省份入档映射与预期 MAE 带;8 项优先级排序 | 文档交付;合计预期:系统中位 I_cfg 65.8→~50-55,A/E 档 ≤35、A 档多区 ≤10-15;序 1(双轨标签)/序 2(NEMED DUID)/序 3(集合 NWP)为地板级杠杆 | FD-10 | `cc:方案就绪待实施` |
| FD-12 | 路线图落地序 1+6:双轨标签(`fuel_*_phys` 自动记录)+ C 档包(evening 加权+solar 界)+ 度时数通道(HDH/CDH) | **双轨 PASS**:UK_14 标签噪声 −35%(65.9→42.5)、UK_13 −16%、UK_12 −8%,CISO 干净(−1%)——D 类在协议级变简单;**C 档包 REJECTED**(UK_11 +15,晚峰加权伤水平);**度时数 NEUTRAL 保留**(I_cfg p=0.98,UK_08 −8.8)。`fuel_decomp_fd12_verdict.md`;fut_exog 扩至 12 通道 | FD-11 | `cc:完了` |
| FD-13 | 数据轨道落地:价格(WB 粉红表+FRED,辖区映射 z-score 1 月滞后)+ 阵风/气压(ERA5,950hPa 探测不可用已替换)+ 数据层集成(wx 7/fut_exog 14)+ NEMED 核查(venv 缺失,遗留) | **信号真实幅度小**:I_cfg 72.32→71.57(win 62%,SA1 −2.5/NSW1 −1.1/BPAT −1.0),p=0.32;I_0/I_+ 中性;通道保留默认。`fuel_decomp_fd13_verdict.md`;全量 145 对(含双轨)后台运行中 | FD-12 | `cc:进行中(全量)` |
| FD-14 | NEMED DUID 数据工程:AU 逐燃料提取器(帧内自分类:名字连接+强度带+夜间测试)+ fuel_shares_au.json + **两个深层 bug 修复**(AU 本地时 vs UTC 假设:solar↔clearsky −0.697→+0.976;DST +1h)+ 确定性风电路由(wind≥0.25→聚合路径) | **PASS**:提取验证 rs 相关 1.000、物理重建 MAE 6-14;I_cfg 快速集中位 **71.57→67.77(win 75%,p=0.074)**,I_0 Spearman 0.277→0.333;NSW1 −10.8、SA1 −6.8;VIC1 残差记录。`fuel_decomp_fd14_verdict.md`;全量 145 对完成:I_cfg 62.75(p=7.1e-11)、I_+ 46.47 首胜 legacy ZS+(win 82%) | FD-13 | `cc:完了` |
| FD-15 | EIA-930 日前负荷通道:提取(`extract_eia_demand.py`,MAPE 5.5%)+ 集成 + 逐窗去均值修正 | **NEUTRAL**:z 版形态 +0.05 但水平 +3(p=0.042);去均值后中性(win 50%,p=0.91)——日历已携带负荷周期形态。保留(部署合法)。**全量最终(FD-14 态):I_cfg 62.75(win 75%,p=7.1e-11)、I_+ 46.47 首胜 legacy ZS+(win 82%,p=5.2e-16,equalizer 破解)**。`fuel_decomp_fd15_verdict.md` | FD-14 | `cc:完了` |
| FD-16 | 极端天气归因 → 风 regime 处理:ERA5 降水 29 区(`download_precipitation.py`)+ 归因分析(`extreme_weather_analysis.md`:Idalia 阵风 z+5.1 而 FPL CIF 反稳 0.64×;UK_01 热浪 27×;波动峰值在风电份额过渡带 [0.15,0.3])+ wx 8→10(regime24/tend6 因果)+ fut_exog 15→17 + 风参考干旱锚定 + **_wn 轨道真 bug 修复**(wcf 通道此前未随扰动重算) | **NEUTRAL(诚实负结果)**:smoke 16 对 I_+ p=0.49;机制诊断(4 风电区 × lull 三分位)两条通路均 ±1 内——336h rs 历史已含干旱信息(冗余),且 lull 残差本质是 NWP 预报技巧(退出时机),架构不可弥补。保留(归一化更诚实 + _wn 正确性修复)。`fd16_wind_regime_verdict.md` | FD-15 | `cc:完了` |
| FD-17 | **风速单位 bug 修复**(Open-Meteo km/h 被当 m/s 喂 IEC 曲线:39-42% 小时误读为切出 cf=0)+ **VIC1/SA1/NSW1 风电场容量加权天气**(`download_farm_weighted_wind.py`,41 场 7 GW;wind_share R² 0.09→0.48/0.15→0.48/0.08→0.33)+ 路由 τ 0.25→0.45(参数化 + eval flag) | **PASS(最大单步提升)**:**I_cfg 62.75→50.15(win 83%,p=3.9e-20)**;**I_0 53.39→45.66 首次大幅胜 legacy ZS(50.72,−5.1)**;I_0_phys 43.14、I_cfg_phys 46.22。区级:UK_10 I_cfg 72→34、UK_17 105→70、VIC1 137→100。代价:I_+ +0.6(vs legacy ZS+ 由 +0.33→−0.27)、NSW1 残留 −8、UK_05 种子不稳。`fd17_wind_unit_farmblend_verdict.md`;官方文件 `fuel_decomp_eval_full_fd17.json` | FD-16 | `cc:完了` |
| FD-18 | 碳流感知 imports-EF(CFCG 思想的可部署形态):`imp_ef` 头(config⊕日相位,±0.9,零初始化),覆盖法国核电 50↔荷兰气电 450 源结构摆动;中国侧=省间月度送受电(交易中心/中电联公开)×送端省结构的流加权 EF | **PASS(小)+机制证实**:6 高 imports UK 区 × 5 seed,I_0 40.15→**39.14**(win 19/30,p=0.009);_phys 轨道退化(p=5e-7)= reported 标签内嵌真实进口记账而静态 250 重建缺失的签名。UK_14(imports 34%):I_cfg 52.5→48.3、I_0 50.6→45.4。保留通路,不重跑全量(仅 UK 激活,全局 ~−0.2)。`fd18_imports_ef_verdict.md` | FD-17 | `cc:完了` |
| FD-19 | 水电主导路由(用户发现 BPAT I_0 直线):可调度水电日内跟负荷(实际 0.60→0.79)但基荷头建模为慢变水平 → 燃料路径形态/水平双崩;`route_fuel ×= σ(20(0.5−hydro_cfg))`,基准内仅 BPAT(0.713)触发 | **PASS**:BPAT I_cfg 46.7→**16.4**、I_0 16.5→**9.3**、I_+ 6.2;冷模式聚合头(config+日历)直接抓负荷形态。官方更新 `fuel_decomp_eval_full_fd19.json`。中国:西南水电省(云/川/青)同走聚合路径。`fd19_hydro_router_verdict.md` | FD-18 | `cc:完了` |
| FD-20 | 结合 actual 的误差归因(11 区残差 dump):冷模式水平偏差=最大剩余源(CISO bias −67.5 而 Spearman 0.717;月度 −52→−76 加深;oracle 逐月去偏上限 11 区 −25%)→ 学习式季节头(回退)+ monthly-config 全量重测(全月=已发布统计口径) | **双模式定案**:学习头 NEGATIVE 已回退;monthly 全量 **I_0 45.66→42.78(p=0.004,首次低于 PatchTST 监督参照 43.50)**;I_cfg 52.30(n.s.)双向分化 —— 季节漂移区 CISO −24/NSW1 −11/UK_11 −11/QLD1 −8,易区 NYIS +7/PJM +5;λ=0.5 收缩与漂移选择规则均无效(corr −0.14)。官方=年度模式不变(50.15/45.66);**部署模式(中国)= 月度接口**,风光季节大省收益 −8~−24。`fd20_seasonal_level_verdict.md` | FD-19 | `cc:完了` |
| FD-21 | 冷模式形态缺失(用户指出 BPAT/UK_16/UK_01 的 I_cfg 曲线):聚合路径冷模式无形态发生器 → τ=1.1 三区探针 + 层级翻转实验 | **路由天花板确认**:UK_16 错路由(I_0 37.9→27.4 可得)但单阈值无法修复(SA1 0.561 要聚合 vs UK_16 0.544 要燃料);BPAT 层级翻转 16.4→46.6 大败已回退(平线=该层级最优);UK_01 极端风电区两路皆无冷形态(Sp≈0)。后续:按结构类别的路由表 / UK farmblend / NWP 集合。`fd21_cold_shape_routing_verdict.md` | FD-20 | `cc:完了` |
| FD-22 | UK 苏格兰 farmblend(UK_01 17 场含 Moray Firth 海上 1.5 GW、UK_16 32 场、UK_02 11 场;R² 0.168→0.383/0.488→0.610/0.458→0.585)+ 部署路由表(UK_01/02/16→燃料路径;SA1 保持聚合)+ 训练期选路实验(四区全选反 = 路由赢家是季节函数,负结果留档) | **PASS**:UK_02 I_cfg 19.5→**16.0(达标<20)**、UK_16 34.3→**28.3**(残差 −26 为 UK API 记账标签噪声)、UK_01 49.4→**41.9≈实证地板**(I_+ 34.3);全局 I_cfg 50.15→**49.79**。SA1 燃料冷偏置 +56 非季节成因(开放问题)。官方 `fuel_decomp_eval_full_fd22.json`。`fd22_uk_farmblend_route_verdict.md` | FD-21 | `cc:完了` |
| FD-23 | **校准有效排放因子**(数据工程解标签记账噪声):训练段岭回归反解各源有效 EF(UK API 互联进口强度/DUID 漂移/EIA 口径),向经典收缩 λ=15,clip [0,1400];上限分析:真值份额残差 UK_14 59.6→27.1、NYIS 19.3→2.5;UK_14 反解 imports EF=137(法国互联混合,物理自洽) | **全面 PASS**:I_cfg 49.79→**47.98**(win 63%,p=0.0013)、I_0 45.66→**43.05**(win 67%,p=2.3e-05)、I_0_phys 39.35;区级 UK_14 −7、UK_13 −3.3、FPL −2。λ=5/15 探针定 15。中国同款可搬(月度结构+年度官方因子联合估计)。`fd23_calibrated_efs_verdict.md`;官方 `fuel_decomp_eval_full_fd23.json` | FD-22 | `cc:完了` |
| FD-24 | 部署栈全叠加(monthly × 校准 EF × 路由表,145 对)+ "MAE<20"严格可达性判定 | I_cfg 47.98→**46.42**(p=0.041 边际;水平修复重叠);**Oracle 分解:完美水平+当前形态的中位=29.5 → 中位 20 低于信息地板与监督上限(43.5),数学不可达**;硬尾=VIC1 81/UK_09 50/NSW1 63 形态地板(NWP 技巧/标签)。<20 俱乐部 5/29(BPAT 14.4 水电、PJM/FPL/NYIS 17-20 煤电平稳、UK_02 15.7),水平修后上限 ~9/29。**中国省级:<20 对水电省(~15)与煤电平稳省(17-20)是可达目标**。`fd24_goal20_floor_verdict.md`;`fuel_decomp_eval_full_fd24.json` | FD-23 | `cc:完了` |
| FD-25 | NSW1 DUID 重标注尝试:全年缓存重建诊断成功(Guthega 水电 121GWh/Shoalhaven 抽蓄 109GWh/BESS 40GWh 混入 wind 桶,反相关污染);夜占比规则设计即否决(真风场 night 同为 0.35-0.45);天气相关性规则(小时/日尺度两版)均过度重标(0.176→0.099/0.095)—— NSW 场表仅覆盖 1.9/3.5 GW,参考失真无从校准 | **NEGATIVE(已完全回退,数据核对与 FD-14 逐区一致,官方 FD-23 不受影响)**:存储混桶定性为残留(~0.4-4% 份额),待 AEMO 注册表可得时做"注册表优先+相关性兜底"两段式。教训同 FD-16:分类器上限=参考数据完整性。`fd25_duid_relabel_attempt_verdict.md` | FD-24 | `cc:完了` |
| FD-26 | 结构理解驱动(用户指令):耦合矩阵(UK 家族 gas=风残差 −0.6~−0.7 但天气质心;QLD1 风 R²=0.000)→ **GB 国家风场混合**(54 场 16.9GW,UK 全国天气同调,无专属表区共享)+ QLD1 专属表(北昆 12 场,R² 0.000→0.422)+ UK_10 近海专属表(R² 0.635→0.823,GB 混合反而伤它)+ loader 回退接线 | **大胜(第三次验证同一规律)**:全量 145 对 **I_cfg 47.98→43.00(win 81%,p=1.4e-14)、I_0 43.05→39.15(p=7.2e-14)**、I_+ 46.90 追平 legacy;区级 UK_17 −15.7/UK_07 −13.0/UK_08 −10.8/UK_12 −10.1/QLD1 −4.6;代价 VIC1 种子 −8~−14(donor 分布变化)。官方 `fuel_decomp_eval_full_fd26.json`。`fd26_gb_blend_verdict.md` | FD-25 | `cc:完了` |
| FD-27 | **US 场加权**(第四次应用):ERCO 25 场 11.6GW(西德州/潘汉德尔/南德州三集群)、MISO 12 场 6.3GW(爱荷华/明尼苏达平原)、CISO 5 场 4.9GW(Tehachapi/Solano);R² ERCO 0.211→**0.714**、MISO 0.167→0.628、CISO 0.051→0.616;fetch 改 curl(urllib TLS 被网络路径切断)+重试 | **大胜**:ERCO I_cfg 65.8→**37.4(−28.4)**、MISO 47.4→**30.6(−16.8)**、CISO 71.2→70.3/I_0 43.0→40.4;子集 win 14-15/15(p≤6e-4)。中期合并官方 `fuel_decomp_eval_full_fd27.json`:**I_cfg 43.00→40.46、I_0 39.15→37.70**;全量确认跑进行中 | FD-26 | `cc:完了` |
| FD-28 | **策展机组注册表**(FD-25 失败的正确解法):四区默认桶逐一诊断 → VIC1 的 MURRAY = Snowy 水电 **3.1TWh(州发电 6%,风桶虚高 1/4)**、QLD 的 Wivenhoe 抽蓄 350GWh(夜发 0.44 完美伪装)、NSW 的 Guthega/SHGEN/6×BESS;公开身份的手工覆盖表(非拟合规则)+ 电池 token | **标签正确性 PASS + VIC1 大改善**:风份额 VIC1 0.320→0.245,R² 0.477→**0.698**,I_cfg 111→101(sp 0.20→0.27);全量确认 **I_cfg 41.72→40.94(p=0.078)**、I_0_phys p=0.036;NSW1 +8 为 donor 波动(自身标签几乎未变)留档。官方 `fuel_decomp_eval_full_fd28.json` | FD-27 | `cc:完了` |
| FD-29 | **多年数据全链条**(用户指令):EIA-930/UK API/场表 2022+2024 全量下载与提取(网络层全面切断 urllib TLS → 四个下载器 curl 化);新协议行:训练 2022-2024、测试 2024Q4 | **结构性发现**:I_+ **38.16(−8.7,追平并低于 persistence 4.2)** —— 多年训练对遥测层明确大赚;I_cfg 61.3 暴露三年均值 config 双重过期(结构漂移+季节),修复=monthly(FD-29b 组合臂启动中)。2023 官方不变。`fd29_multiyear_verdict.md`;`fuel_decomp_eval_full_fd29_multiyear.json` | FD-28 | `cc:完了` |
| FD-29b/c | 多年协议补丁:月度表多年模式下三个年份同月平均稀释结构漂移 → 改"最近年"语义(= 官方统计按年发布);b/c 两臂对比 | I_cfg 61.3→60.1→**58.8**(月度语义修复 −1.4);冷模式多年退化的主因排除月度过期,定位到多年特有失真(需求通道 22/24 置零/EF 校准窗漂移/气候均值混合)留档;I_+ 38.2 稳定。19 数据测试 ✓ | FD-29 | `cc:完了` |
| FD-30 | "MAE<20" 收官:2023 协议 × monthly × 当前全栈(首次组合)→ monthly 已被场加权+校准 EF 吸收(中性偏负 p=0.76,UK_16 种子回归)→ 年度配置(FD-28)为基准最终态 | **<20 俱乐部 4/29**(UK_02 14.3/NYIS 18.2/BPAT 18.5/PJM 19.6)+ 20-25 可达集 4/29(ISNE 地板 9.8);中位 20 = 数学不可达第三次确认(地板 29.5/监督 43.5/当前 40.94);**中国省级映射:水电省 ~15-19✓、煤电平稳省 18-22✓**。会话总账:I_cfg 62.75→40.94(−35%)、I_0 53.39→37.17、I_+(多年)38.2。`fd30_goal20_final_verdict.md` | FD-29b/c | `cc:完了` |
| FD-31 | 源偏置校准探针(余量排名 UK_09/NSW1/VIC1/CISO 冷偏置 −52~−70 触发;并行流已建好的 `--source-bias-calibrate` 机制:donor holdout 偏置 config 相似度加权迁移,部署安全) | **小/中性,不进官方**:win 35/40(p=1e-5)但估计偏置仅 −4~−8 vs 实际 −67~+70 —— 剩余冷偏置为目标期特有(Q4 漂移)不可迁移;UK_15/GB/CISO −6,NSW1 +3.7 反向。**水平类修复正式宣告榨干**(FD-24/30/31 三重互证)** | FD-30 | `cc:完了` |
| FD-32/33 | 收缩度探针(λ=0.5→0.8)+ 多年需求通道修复(提取器按年拆分 ×8 区×3 年,loader 多年 glob)+ CISO 多年瞬态定性 | **λ=0.8 NEGATIVE**(70.2→73.4,λ=0.5 即最优);**多年需求 MIXED**(VIC1 −7.6 真/UK_09 +8.8 donor 噪声;正确性修复保留)。水平/收缩/需求三维度全部关闭,多年冷模式残余=donor-shift 噪声主导 | FD-31 | `cc:完了` |
| FD-34 | **冷模式月度锚**(绿橙差距分解的直接产物:遥测价值=水平 CISO/NSW1 +18 + 形态;I_0 锚定从未移植冷模式):config[0] 在 monthly 接口下=公开月度可再生份额 → (1−份额)·ef_nr 锚水平,零初始化门 | **目标类大胜**:CISO −14.6(预测 17.6)、NSW1 −12.4(预测 18.1),子集 I_cfg −7.2(p=3.2e-4);全量中位 40.94→40.04(p=0.36)/均值 −1.25(易区被 monthly 抵消)。**基准官方维持 FD-28;部署模式 v2 定案 = monthly × 冷锚**(中国接口) | FD-32/33 | `cc:完了` |
| FD-35 | **尖峰跟随诊断**(用户直觉,`diag_spike_following.py`):>30 区主病=**过平漏尖峰**(离散比 0.35-0.74,7/10 区;集中比 1.2-1.4),过度跟随 3 区(UK_05 2.36/CISO 1.48);+ **动态残差头**(幅度界 220)并入部署栈 | 过平家族 I_cfg −2~−4(win 28/30,p=1.3e-7),伤害检查过度家族也赢(CISO −3.8/UK_11 −2.4,仅 UK_05 +3.2);**机制校验:MAE 降但离散不变 → 修的是逐窗偏移非尖峰;欠离散=调度/停机事件无公开日前签名,定性为信息边界**。部署 v3 全量:**I_cfg 40.04→39.69(win 62%,p=1.5e-5)** | FD-34 | `cc:完了` |
| FD-36 | "每区→20"终局判定:最优模式 I_cfg vs 逐区监督上限(PatchTST) | **4 区已<20**(UK_02 14.3/NYIS 18.2/BPAT 18.5/PJM 19.6);2 区可攻未达(FPL 21.2 形态/ISNE 22.8 进口缺数据);**23/29 区监督上限本身>20**(VIC1 92/UK_09 78/SA1 52);**11 区零遥测已击败监督模型**(VIC1 −12.5/UK_10 −17.7)。目标在中国省级成立(水电/煤电平稳省对标 18-20) | FD-35 | `cc:完了` |
| FD-37 | 数据工程三探针(用户"还有空间"挑战):AU ef_nr 核对(NEMED 隐含值,≤0.2% 无 bug)、冷锚近锚定初始化(p=0.96 中性+VIC1 不稳已回退)、**EIA-930 interchange 进口通道**(全新提取 8 区×3 年;去均值 MW 变体 p=0.47、进口份额变体 p=0.16 趋势正不显著;保留份额通道,n_exog 18) | **全部中性,通道按物理原则保留**(中国省间交换同构)。基准内数据工程穷尽结论与 FD-24/30/31/32/35 五重互证;剩余=外部数据(NWP/调度/注册表) | FD-36 | `cc:完了` |
| FD-39 | 夜间持续冲刺(/goal):TTA、trust 门控、路由复位三轮全量;**定因一次协议事故**:CLI 默认 `--wind-route-tau` 在 FD-36/37 期间 0.45→1.1 静默漂移 → 全燃料路径(VIC/UK_05/BPAT/UK_16 大幅受益,SA1 合成标签区失保护 +25) | **官方刷新 fd39e:I_cfg 38.69 / I_0 37.44 / I_+ 47.03**(对 fd35 文件 I_cfg −1.7,配对 n.s.;UK_05 48.4→40.3、UK_16 I_0 33.3→20.4 自愈、SA1 59.9→85.5 代价);fd39f(路由复位+表)SA1 57.6 保护版归档;**负结果:TTA 中性、trust 门控无增益、评估期路由强加仅 UK_16 有效(FD-22 收益需训练期路由)**。方法学:CLI 默认值=协议参数必须进 verdict;形状/默认值改动必须重跑官方。`fd39_night_verdict.md`;`fuel_decomp_eval_full_fd39e/f.json` | FD-37 | `cc:完了` |
| FD-40 | 种子集成部署模式:同目标 5 seed 预测平均(纯部署方差缩减,零信息增量;同窗口同官方种子) | **29/29 区全部改善,无一受损**:I_cfg 中位 38.69→**37.46**(−1.23)、I_0 37.44→**37.04**;VIC1 −3.1/UK_05 −3.0/MISO −1.8/**NYIS 19.6 进入 <20 俱乐部**;SA1 无效(−0.1,水平偏置非种子噪声,交叉确认 §2)。部署定案:5-seed 集成 + 全燃料路径 + 合成份额省路由保护。`fd40_seed_ensemble.json`;`probe_seed_ensemble.py` | FD-39 | `cc:完了` |

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

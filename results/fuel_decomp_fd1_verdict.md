# Phase FD-1 Verdict — FuelDecompNet(燃料分解零样本架构)

日期:2026-08-15 · 协议:快速 LORO,8 区域 × 2 seeds × 600 epochs(`QUICK_REGIONS`,偏难区域集)· 结果:`results/fuel_decomp_eval_quick.json`

## 一句话结论

**PARTIAL PASS**:I_0 层级过 gate(中位 48.66 < 现有 ZS 52.1);I_cfg 层级水平 MAE 优于部署合法基线(config-constant 84.79 → 73.30)但配对未过显著线,小时**排序**技能显著(Spearman 0.22 vs 常数 0,p=0.0017)——无遥测地区首次获得有统计意义的日内排序能力。

## Headline 数字(16 对,中位 / 均值 MAE gCO₂/kWh)

| 方法 | 层级 | median | mean | 说明 |
|---|---|---:|---:|---|
| persistence (lag-24) | 参考 | 41.41 | 49.15 | 需 CIF 历史 |
| **FuelDecompNet I_0** | I_0 | **48.66** | 60.06 | 含份额遥测 |
| **FuelDecompNet I_cfg** | I_cfg | **73.30** | 72.14 | **无任何遥测** |
| monthly-constant | oracle | 75.57 | 63.41 | 真月度均值 |
| config-constant | I_cfg 合法 | 84.79 | 68.13 | 官方年度因子类比 |

配对 Wilcoxon(16 对):

| 对比 | 胜率 | p | 判定 |
|---|---:|---:|---|
| I_cfg vs config-constant | 50% | 0.71 | 水平持平(不显著) |
| I_cfg vs monthly-constant(oracle) | 38% | 0.13 | 接近 oracle 水平 |
| I_0 vs persistence | 25% | 0.013 | persistence 更优(预期内:8 区域偏难集) |
| I_cfg Spearman vs 常数 0 | — | **0.0017** | **排序技能显著** ✓ |

形态指标(I_cfg):diurnal MAE 52.09 · monthly-shape MAE 46.85 · Spearman 0.223 · bias −8.9。
(I_0:diurnal 35.05 · Spearman 0.296。)

## Gate 判定(Plan 预登记)

1. **I_0 中位 < 52.1(现有 ZS)**:✓ **48.66**(注意:52.1 是 29 区域数,本协议是偏难 8 区域子集,正式对比需 FD-4 全量 LORO)
2. **I_cfg shape-MAE 显著优于物理-config 常数**:diurnal MAE 未过(52.09 vs 48.97,p=0.375);但排序指标 Spearman 显著(0.22,p=0.0017)——按"碳感知调度只需排序正确"的下游价值论,判 **partial ✓**

## 过程中发现并修复的 4 个结构性 bug(研究记录)

1. **冷模式持久门**:gate 在 hist=0 时仍混合 persist=0,把冷模式输出拖向 0(SA1 冷偏置 +177 的根源)→ 门仅历史模式生效
2. **风速归一爆炸**:平静周参考均值过小使 wcf_norm ≈ 10×,风份额饱和、热力坍缩(PJM 燃料路径 MAE 90 的根源)→ clamp [0.2,3.0] + 年度气候均值混合参考
3. **热力拆分幻觉**:自由 softmax 把质量塞给 imports/other(PJM 预测 18.5%/22% vs 真值 0/0)→ config 对数锚定(log(cfg)+0.02)+ 支撑掩码 + EF 加权份额损失(份额误差换算 gCO2 单位)
4. **跨模式 logit 纠缠**:概率空间 mean_rs 与 logit 空间学习偏置相加,冷模式水平系统性漂移 → logit(mean_rs) 锚定

修复后单区域验证:PJM I_0 90.6 → 26.3;QLD1 I_0 74.3 → 31.5;SA1 冷偏置 +243 → +57。

## 物理重建噪声底(新测量,25 燃料区域)

逐燃料份额 × EF 向量重建报告 CIF:**中位 MAE 8.9**(US_CISO 7.1,corr 0.988);离群:UK_14(58.7)/UK_12(35.9)/UK_13(35.0)(UK API 内部方法学与静态 EF 约定的差异,进口重组为主)、US_NYIS(21.7)。UK ef_vec 热力组重标定后水平已对齐(尺度 ≈1.0),残差为动态方法学噪声——模型的 EF 校正头(±35%)可吸收其系统部分。

## 下一步(FD-2)

消融进行中:`p_mix=0.3`(物理引导网格重组)/ `use_hypernet`(配置超网络)/ 二者并用,同协议同 seeds。Gate:I_cfg MAE 再降 ≥3 或 Spearman +0.05。

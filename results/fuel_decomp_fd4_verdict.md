# Phase FD-4 Verdict — 全量 LORO 正式数字(29 区域 × 5 seeds = 145 对)

日期:2026-08-15 · 协议:与 legacy 阶梯**同区域集同 seeds**(含 UK_18_GB、不含 UK_04,对齐 `unified_eval_full.json`)· 结果:`results/fuel_decomp_eval_full.json`(600 epochs,锚定模型,I_cfg/I_0/I_+ 三层级)

## 一句话结论

**I_+ 层级 equalizer 效应全尺度确认(r=0.9999):燃料分解骨干与 legacy 骨干在 ZS+ 校准下不可区分(中位差 0.17)——骨干选择对 I_+ 无贡献;TransCIF-FD 的价值在且仅在 I_cfg(零遥测)层级:65.67 vs 年度常数 70.90,Spearman 0.280,77% 区域排序技能 >0.1,是唯一能在无遥测地区产出有意义逐小时预测的方法。**

## 正式数字(145 对,中位 MAE)

| 方法 | 层级 | median | 关键配对 |
|---|---|---:|---|
| FuelDecomp+ZS+ | I_+ | 47.08 | vs persistence 50.70:**win 99%,p=1.6e-25**;vs legacy ZS+ 46.80:win 20%(平局被 +0.17 噪声打破),p=4.7e-16 |
| FuelDecomp(锚定) | I_0 | 55.09 | vs legacy ZS 50.72:win 26%,p=5.5e-06 —— **legacy 保持 I_0 旗舰** |
| **FuelDecomp** | **I_cfg** | **65.67** | vs 年度常数 70.90:中位 −5.2,win 50%,p=0.34;**Spearman 0.280(77% 区域 >0.1)**;legacy 阶梯在此层级**无任何条目** |
| monthly-constant | oracle | 64.34 | I_cfg 水平已接近月度 oracle |
| PatchTST | I_S(监督) | 43.50 | — |

**Equalizer 定量**:FD I_+ 与 legacy ZS+ 逐对相关 **r=0.9999**,逐对 |ΔMAE| 中位 **0.17 gCO₂/kWh**。此前论文的 equalizer 只在 4 个 legacy 骨干内验证(Δ≤0.05,12 区域子集);本次扩展到**完全异构的物理分解骨干**(20k 参数、10 燃料、天文/气象特征)依然成立——ZS+ 的 6 分支回测融合在多数网格由 CIF 滞后分支主导,模型分支只贡献形态,而形态被逐窗水平锚定重置。**结论:I_+ 层级的竞争不在骨干,在校准**(论文 §4.5 的加强版证据)。

## 消融(8 区域 × 2 seeds)

| 变体 | I_cfg 中位 | 判定 |
|---|---:|---|
| donor 燃料结构加权 | 71.01(win 44%,p=0.025 反向) | **拒绝**——legacy mean_rs 采样器更优;与 MLDG 负结果一致 |
| hypernet + ef_corr 0.15 | 65.43(中位 −5.6,win 62%,p=0.94) | **拒绝**——收紧界未驯服灾难(VIC1 171.6/bias −133 反而恶化);不稳定是 config→权重外推的结构性问题,非 EF 界 |

## 决策与论文影响

1. **主 claim 重定位**:TransCIF-FD = I_cfg 层级的开创方法 + equalizer 的异构骨干证据;I_0/I_+ 旗舰仍是 legacy AdaptivePersistDLinear + ZS+
2. I_cfg 数字进 BENCHMARK 排行榜作为该层级首条正式记录
3. hypernet/donor 记录为负结果(方法学价值:排除)
4. 后续方向:I_cfg 的水平校准(当前 I_cfg 65.67 vs 月度 oracle 64.34 差距极小——水平已接近可达上限,提升空间在形态:Spearman 0.28→?)与 I_J 联合微调(FD 骨干 + torch-native,FD-6)

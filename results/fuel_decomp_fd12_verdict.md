# Phase FD-12 Verdict — 路线图序 1(双轨标签)+ 序 6(C 档包)+ 度时数

日期:2026-08-15 · 实验:D 簇+C 类 6 区域×2 seeds(`fuel_decomp_eval_dualtrack.json`)、C 档包消融(`fuel_decomp_eval_cclass.json`)、12 通道干净对照(`fuel_decomp_eval_deg12.json` vs FD-6 `fuel_decomp_eval_shape.json`)

## 一句话结论

**双轨标签 PASS(已内置)/ C 档包 REJECTED / 度时数 NEUTRAL(保留)**:评估现在对燃料区域自动并行记录"vs 报告 CIF"与"vs 物理真值(份额×EF)"两条轨道——D 类的标签噪声在协议级被隔离(UK_14 **−35%**、UK_13 −16%、UK_12 −8%);晚峰加权+solar 界放宽组合伤害 UK_11(+15)与形态,拒绝;度时数通道(HDH/CDH)I_cfg 中性(p=0.98)但 UK_08 −8.8,保留。

## 1. 双轨标签(路线图序 1,PASS)

D 簇 + 对照 × 2 seeds,同一预测双真值:

| 区域 | I_cfg vs 报告 | vs 物理真值 | 标签噪声占比 |
|---|---:|---:|---:|
| UK_14_South_East | 65.9 | **42.5** | **−35%** |
| UK_13_London | 62.1 | 51.8 | −16~17% |
| UK_12_South_England | 71.0 | 65.0 | −8% |
| UK_08_West_Midlands | 103.0 | 97.9 | −5% |
| UK_11_South_West | 68.9 | 66.6 | −3% |
| US_CISO | 106.8 | 107.5 | −1%(干净) |

与 FD-10 分解实验一致;**论文/排行榜建议 D 类双轨并报**(API 计账轨 + 物理轨),`fuel_*_phys` 键每次评估自动生成。

## 2. C 档包(路线图序 6,REJECTED)

`--evening-weight 2.0 --solar-mod-bound 0.6` vs 同代码基线:

| 区域 | I_cfg | diurnal | 判定 |
|---|---|---|---|
| UK_11_South_West | 68.9→**84.6**(+15.7) | 46→68 | 严重伤害 |
| UK_08(对照) | +1.0~1.3 | +3 | 轻伤 |
| US_CISO | 持平 | +1.5 | 无益 |

机制:晚峰 ×2 加权扭曲水平校准(晚峰 CIF 高→全天高估),solar 界放宽放大天气噪声;C 类在数据集仅 2 区,且 CISO 的 solar 头本已工作良好(天文主导)。**默认保持 evening=1.0 / bound=0.4**。C 档的鸭子曲线误差主项(晚峰热电爬坡)需要的是 E 档的价格/负荷通道而非加权。

## 3. 度时数通道(NEUTRAL,保留)

12 通道(HDH/CDH 入 fut_exog)干净对照(8×2,vs FD-6 10 通道):

- I_cfg:71.76→72.32(win 50%,p=0.98)中性
- I_0:45.68→45.34(UK_08 112→**103**,VIC/PJM 微改善,NSW +1.5)
- 保留:部署合法(再分析温度)、无系统伤害、UK_08 类显著受益

## 决策

1. 双轨标签为默认行为(所有后续运行自动带 `_phys` 轨)
2. C 档包两个 flag 保留但默认关闭(复现:`--evening-weight 2.0 --solar-mod-bound 0.6`)
3. 度时数保留为 fut_exog 标准通道(n=12)
4. 路线图更新:序 6 完成(负结果),C 档后续走价格/负荷通道(与 E 档合并)

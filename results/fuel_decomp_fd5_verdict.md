# Phase FD-5 Verdict — TransCIF 经验移植:I_0 水平锚定 + ZS+ 校准接入

日期:2026-08-15 · 协议:同 FD-1(8 难区域 × 2 seeds × 600 epochs)· 结果:`results/fuel_decomp_eval_anchor.json` vs 基线 `fuel_decomp_eval_quick.json`

## 一句话结论

**PASS(核心 claim) + PARTIAL(次级 claim)**:把 TransCIF 历程中验证过的两大机制移植进 TransCIF-FD 后,**I_+ 层级 16/16 全胜 persistence(36.81 vs 41.41,p=3.1e-5)**,在刻意选难的 8 区域集上首次让燃料分解模型全面超过标准预测基线;I_0 锚定中位 −3.5(48.66→45.13,UK_08 −23)但配对 p=0.10;I_cfg 不受影响(p=0.94,锚定冷模式关闭,符合设计)。

## 机制与来源(TransCIF 探索经验 → FD 移植)

| 仓库结论(证据) | 移植实现 |
|---|---|
| ZS+ branch-0 水平锚定是主力(drop-one verdict = `ZS_PLUS_WRAPPER`;52.1→46.88) | `FuelDecompNet._anchor_correction`:模型出**形态**、观测 rs 流出**水平**((1−rs̄48h)·ef_nr),`anchor_gate` 门控(sigmoid 偏置 1.5 起步近锚定),仅历史模式生效——**I_0 合法,不需 CIF 历史** |
| ZS+ 校准管线 29/29 胜 persistence(legacy) | `make_zs_plus_share_fn` + `zs_plus_predict` 接入评估,燃料模型作 anchor branch(I_+ 层) |

## 数字(16 对,中位 MAE / 配对 Wilcoxon)

| 方法 | 层级 | median | 关键对比 |
|---|---|---:|---|
| FuelDecomp+ZS+ | **I_+** | **36.81** | vs persistence 41.41:**win 16/16,p=3.1e-5**;vs 同次 I_0 45.13:p=3.1e-5;Spearman 0.412(vs I_0 0.292,p=3.1e-5) |
| FuelDecomp(锚定) | I_0 | 45.13 | vs FD-1 基线 48.66:中位 −3.5,win 50%,p=0.10(UK_08 114→91、VIC1 持平;QLD1 +3 小回退) |
| FuelDecomp(锚定) | I_cfg | 71.01 | vs 基线 73.30:p=0.94(锚定冷模式关闭;训练端让模型卸下水平负担专注形态) |
| persistence | 参考 | 41.41 | — |

逐区域 I_+ vs persistence:**16/16 胜**(VIC1 98.5/116.8、SA1 60.5/68.1、NSW1 46.5/53.7、UK_08 77.9/81.5、UK_02 21.6/22.1、QLD1 27.0/29.1、PJM 14.1/15.6、BPAT 6.2/6.3)。

## 与 legacy 阶梯的关系

legacy ZS+(n=145,全区域)中位 46.80;本次 fuel+ZS+ 在**更难的 8 区域子集**上 36.81(同子集的 legacy 数字会更高)。正式可比需 FD-4 全量 LORO(29×5)。但结构性改进明确:I_+ Spearman 0.412,形态质量同步提升。

## 决策

1. `anchor_gate` 与 I_+ 评估路径并入默认 FD 流程(单测 11 绿,主套件无回归)
2. 下一步优先级:FD-4 全量 LORO(确立 29×5 正式数字)→ donor 加权升级(燃料结构距离)→ hypernet 水平通路稳定化

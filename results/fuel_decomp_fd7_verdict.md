# Phase FD-7 Verdict — 月度电力结构接口(中国省份部署形态)

日期:2026-08-15 · 消融:8×2 快速协议(`fuel_decomp_eval_monthly.json` vs `fuel_decomp_eval_shape.json`)· 部署演示:`scripts/experiments/demo_cn_province.py`

## 一句话结论

**基准上 NEGATIVE / 部署接口交付**:月度 config 条件化在本基准 I_cfg 上小幅显著变差(MAE 71.76→72.21,win 12%,p=0.01;Spearman 中性)——29 个基准区域的年燃料结构平稳,月度分辨率引入噪声而非信号,且 UK"可再生含核电"定义使月度 mean_rs 失真(UK_08 114→128)。**默认保持年度 config**;月度接口作为 `--monthly-config` 选项保留,配 12×16 月度表 + 1 月发布滞后查询,并交付端到端省份部署演示。

## 月度接口实现(全部部署合法)

输入:过去 12 个月 × 逐燃料发电量(公开统计)→ 转换:

| 接口字段 | 从月度发电量导出 |
|---|---|
| 月度燃料份额 (12×10) | `gen_f(m) / Σ_f gen_f(m)` |
| ef_nr | 火电子结构 EF 加权(煤 980/气 410/油 650,IPCC)+ imports/other |
| mean_rs(m) | Σ(水+风+光[+生物质])份额 —— **由份额导出,不经 rs 遥测** |
| ann_windcf / ann_csi | 再分析天气气候态(Open-Meteo,任意坐标) |
| |lat|/60、has_fuel | 公开 |

预测窗 config = 月度表[m − 1](发布滞后 1 月;7 月预测用 6 月数据)。

## 部署演示(`demo_cn_province.py`,虚拟华北煤主省份)

年结构:煤 66%/风 9%/光 8%/气 4%(mean_rs 0.166,ef_nr 703)。7 月样本周 I_cfg 预测(零遥测):

- 周 mean CIF **497.6**(年度常数口径 586 —— 月度表正确捕捉夏季煤电回落)
- **光伏份额峰 → 当地 13:00,CIF 谷 391.7 随之出现**;煤电份额 0.68(夜)→0.54(午)跟随光伏让路
- 日摆幅 171 gCO2/kWh —— 碳感知调度的可用信号

## 决策

1. `LAMBDA_SHAPE=0.5` + 年度 config 保持默认;`--monthly-config` 保留(中国场景选项)
2. 基准边界记录:静态结构区域集无法验证月度接口的价值;真实省份验证需中国数据(论文 future work / 合作试点)
3. FD-7 候选剩余:solar 晴空调制界放宽、I_J 联合微调

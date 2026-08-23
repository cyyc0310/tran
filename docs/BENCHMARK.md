# TransCIF Benchmark:无遥测区域电网碳强度(CIF)预测基准

> 版本:v0.1(2026-08-15,Phase FD)· 任务族:零样本跨区域 CIF 预测 · 数据:29 区域 × 2023 全年逐小时

## 1. 任务定义

在**一组源电网**上训练的模型,能否预测一个**从未见过、位于不同司法辖区**的目标电网的逐小时碳强度(CIF, gCO₂/kWh)?

**部署动机**:全球多数电网(如中国省级电网)只公开年度排放因子与月度电源结构,没有逐小时 CIF 遥测。接入这些地区的唯一合法接口是配置统计 + 公开外生数据(再分析气象、日历)。

## 2. 信息层级(Information Tiers)

| 层级 | 目标侧输入 | 类比现实 |
|---|---|---|
| **I_cfg**(新) | 燃料份额配置 + 天气(再分析) + 日历/天文,**无任何遥测** | 中国省份:月度电源结构 + 公开气象 |
| I_0 | I_cfg + 实时可再生份额流(336 h) | 公开份额遥测的电网 |
| I_+ | I_0 + 可观测 CIF 历史(ZS+ 校准) | 电网发布历史 CIF |
| I_J | I_+ + 288 h 目标 CIF 标签(联合微调) | 短期本地采集 |
| I_S | 80% 全年本地标签(监督上限) | 传统单区域训练 |

**合法性规则**:每层允许的目标侧输入只有上表所列;源区域(训练侧)可无限制使用其全部数据。所有静态配置量(均值份额、ef_nr、年度气象气候均值)只从目标训练期(前 80%)或公开统计推导,不得接触测试期。

**I_cfg 接口字段**(全部公开可得):`[mean_rs, ef_nr/1000, 10 维年度燃料份额, 年均风电机组容量因子, 年均晴空指数, has_fuel, |lat|/60]`(见 `FD_CONFIG_FIELDS`)。

## 3. 评估协议

- **LORO**(leave-one-region-out):28 源 → 1 目标,29 区域 × 5 seeds = 145 对
- **LOJO**(leave-one-jurisdiction-out):跨辖区迁移(AU⇄UK⇄US)
- 测试窗:最后 20%,L=336 h,H=24 h,stride=24;指标在每个 (区域, seed) 上汇总后报告中位数/均值
- 显著性:配对 Wilcoxon / paired-t / bootstrap CI(20k) + Holm-Bonferroni(`evaluation/stats.py`)

**日前预报双轨道**(未来信息合法性,见 `results/fuel_decomp_fd8_verdict.md`):

| 轨道 | 未来 24 h 天气 | 说明 |
|---|---|---|
| **Track A(主榜)** | 再分析(完美知识) | 与文献可比(EnsembleCI 同口径);天文/日历通道本就确定 |
| **Track B(部署口径)** | NWP 报技能退化:温度 N(0,2K)、风速 ×(1+N(0,20%))、短波 ×(1+N(0,25%)),派生量一致重算(`--weather-noise`) | 实测敏感度:MAE 中位不变、均值 +2-4%(n.s.),Spearman −0.02 |

输入合法性硬规则:目标值(CIF/份额)从不进入输入;历史通道严格取 origin 之前;ZS+ 分支全为滞后量;月度统计 1 月发布滞后。

## 4. 指标

| 指标 | 定义 | 为什么 |
|---|---|---|
| **MAE** | gCO₂/kWh,主指标 | 与既有文献可比 |
| **diurnal MAE** | 逐窗去均值后的 MAE | 纯形态技能,与水平解耦 |
| **monthly-shape MAE** | 预测对"真月度均值"锚的偏差 MAE | 月度水平 + 日内形态的综合 |
| **Spearman ρ** | 逐 24 h 窗预测-真值秩相关(均值) | 碳感知调度只需小时**排序**正确 |
| bias | 符号化水平偏差 | 诊断水平锚定 |
| CRPS / 覆盖率 | 概率预测(可选轨道) | 不确定性量化 |

## 5. 基线(每层至少一个合法基线 + 一个参考)

| 基线 | 层级 | 说明 |
|---|---|---|
| annual-constant | I_cfg | `mean_rs·ef_r+(1−mean_rs)·ef_nr`——**官方年度因子类比**,部署合法 |
| monthly-constant | I_cfg⁺(oracle) | 真月度均值常数,水平上界参照 |
| persistence (lag-24) | 参考 | 需 CIF 历史;标准预测基线 |
| physics-config-constant | I_cfg | `Σ_f config_share_f·ef_f` |
| TransCIF-ZS / ZS+ | I_0 / I_+ | 既有旗舰(`AdaptivePersistDLinear` + ZS+) |
| CarbonCast-ZS | 参考 | 外部跨网格基线 |
| PatchTST-supervised | I_S | 监督上限(41.47 中位,29 区域) |

## 6. 排行榜

`scripts/benchmark/run_benchmark.py` 汇总所有方法输出 `results/leaderboard.json`:

```json
{
  "method": "fuel_decomp_i_cfg",
  "tier": "I_cfg",
  "n_pairs": 145,
  "median_mae": 0.0, "mean_mae": 0.0,
  "median_diurnal_mae": 0.0, "median_monthly_shape_mae": 0.0,
  "median_spearman": 0.0, "median_bias": 0.0,
  "win_rate_vs_annual_constant": 0.0,
  "paired_wilcoxon_p_vs_annual_constant": 0.0
}
```

提交规则:任何方法只要遵守第 2 节合法性规则即可入榜;报告 5 seeds 中位数与配对检验 p 值。

## 7. 当前结果(FD-1 快速协议,8 区域 × 2 seeds)

见 `results/fuel_decomp_eval_quick.json` 与 `results/fuel_decomp_fd1_verdict.md`(Phase FD-1 verdict)。

## 8. 已知边界

- 单年(2023)数据;季节内插值未验证
- UK 数据源的 CIF 由 API 内部方法学计算,逐燃料物理重建残差中位 ~9 gCO₂/kWh(South-East England 进口重组最高 ~59)
- MAE 物理地板:噪声地板中位 52.9 / persistence 地板 50.7(`mae_floor_analysis.json`)——系统级中位数低于 ~30-35 无物理依据
- AU NEM 区域无逐燃料遥测,走聚合回退路径(等价旧物理层)

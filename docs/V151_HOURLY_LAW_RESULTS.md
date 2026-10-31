# V151：回到小时CIF后，完整月信息与可发布信息表现不同

本轮完成36个已查看地区月（PACE/LGEE/SC各12个2025月）、1,152条月份方法评分。拟合形状与残差仅用30个proper源区2024年，校准用另12源2024年；两次源角色划分。所有月份进入共同分母，目标总有效小时PACE8,760、LGEE8,520、SC8,664。比较四种完整概率分布和两种区间校准；CRPS使用全原子加权经验分布前缀和精确计算，五点独立有限和恒等核对通过。

## 主要判断

普通条件重组已经构成强基线。同月完整统计下，max-region校准的宏小时覆盖93.89%/93.97%，最差地区89.35%/89.00%，CRPS45.40/45.46；它恢复了本批困难地区的小时覆盖。相对简单水平残差分布CRPS下降9.71%/8.60%，但相对同预算等权重组只下降1.22%/0.60%，尚未达到对强基线至少5%的要求。区间宽度也没有对全部比较对象统一下降。重组和组成核权重均属常规基线，不能凭此认定原创突破。

按月末排他时刻+30天模拟发布后，pooled校准的普通条件重组宏覆盖94.85%/88.50%、最差92.23%/82.83%；第二划分未达门槛。换max-region后近全覆盖，宽度却达632.71/546.96 g/kWh，而完整同月统计下为313.91/316.58。高覆盖来自校准区间明显扩张；原始概率分布CRPS仍恶化到53.87/54.03，不能用加宽消除这部分分布误差。

PACE仍最难：即便给同月完整组成，pooled校准覆盖81.39%/80.80%，其CRPS约82.44/82.54，明显高于LGEE约19.87/20.15。说明本批目标的困难并非仅由粗统计不及时造成。整月边际分布没有日内小时相位，本轮也不证明小时条件形状已恢复。

校准源的NWMT/GRID驱动最大半径。以PACE第一划分条件重组为例，NWMT的q90校准误差从同月预算30.43升至滞后预算224.43；SC第一划分同样由NWMT驱动。这支持将统计滞后误差与当月调度形状误差单独诊断，不能据目标效果剔除这些校准源。

## 宏小时指标

以下按地区等权；先在地区内按有效小时汇总。CRPS/MAE/区间宽度/Winkler单位均为g/kWh，越低越好；覆盖率是名义90%小时区间的经验覆盖。校准只改变区间，不改变原始预测分布和CRPS。

| seed | 信息预算 | 方法 | 校准 | CRPS | 覆盖 | 最差覆盖 | 宽度 | Winkler |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 41 | asof_30d | level_residual | max_region | 57.255 | 100.00% | 99.99% | 657.93 | 658.16 |
| 41 | asof_30d | level_residual | pooled | 57.255 | 86.66% | 68.69% | 295.32 | 436.96 |
| 41 | asof_30d | recompose_conditional | max_region | 53.872 | 100.00% | 99.99% | 632.71 | 632.95 |
| 41 | asof_30d | recompose_conditional | pooled | 53.872 | 94.85% | 92.23% | 363.26 | 393.76 |
| 41 | asof_30d | recompose_pooled | max_region | 55.107 | 100.00% | 99.99% | 642.12 | 642.33 |
| 41 | asof_30d | recompose_pooled | pooled | 55.107 | 95.29% | 92.67% | 397.10 | 423.31 |
| 41 | asof_30d | unconditional | max_region | 345.749 | 67.48% | 13.03% | 838.48 | 1406.97 |
| 41 | asof_30d | unconditional | pooled | 345.749 | 27.55% | 0.00% | 649.10 | 3104.10 |
| 41 | same_month | level_residual | max_region | 50.276 | 92.51% | 78.17% | 349.07 | 421.11 |
| 41 | same_month | level_residual | pooled | 50.276 | 79.57% | 49.03% | 206.84 | 490.18 |
| 41 | same_month | recompose_conditional | max_region | 45.396 | 93.89% | 89.35% | 313.91 | 346.36 |
| 41 | same_month | recompose_conditional | pooled | 45.396 | 88.83% | 81.39% | 263.15 | 341.53 |
| 41 | same_month | recompose_pooled | max_region | 45.955 | 95.51% | 91.29% | 358.71 | 384.07 |
| 41 | same_month | recompose_pooled | pooled | 45.955 | 89.38% | 82.66% | 298.01 | 371.95 |
| 41 | same_month | unconditional | max_region | 345.749 | 63.88% | 14.52% | 828.42 | 1379.37 |
| 41 | same_month | unconditional | pooled | 345.749 | 27.29% | 0.00% | 647.57 | 3107.57 |
| 83 | asof_30d | level_residual | max_region | 56.417 | 99.45% | 98.34% | 504.94 | 507.58 |
| 83 | asof_30d | level_residual | pooled | 56.417 | 83.62% | 59.50% | 277.75 | 483.79 |
| 83 | asof_30d | recompose_conditional | max_region | 54.033 | 99.95% | 99.84% | 546.96 | 547.44 |
| 83 | asof_30d | recompose_conditional | pooled | 54.033 | 88.50% | 82.83% | 300.01 | 377.13 |
| 83 | asof_30d | recompose_pooled | max_region | 54.973 | 99.95% | 99.86% | 540.01 | 540.45 |
| 83 | asof_30d | recompose_pooled | pooled | 54.973 | 88.10% | 83.12% | 302.96 | 386.59 |
| 83 | asof_30d | unconditional | max_region | 320.281 | 36.25% | 0.00% | 720.37 | 2404.70 |
| 83 | asof_30d | unconditional | pooled | 320.281 | 35.03% | 0.00% | 702.03 | 2740.70 |
| 83 | same_month | level_residual | max_region | 49.734 | 90.59% | 78.16% | 298.89 | 384.10 |
| 83 | same_month | level_residual | pooled | 49.734 | 78.33% | 44.81% | 202.62 | 519.16 |
| 83 | same_month | recompose_conditional | max_region | 45.459 | 93.97% | 89.00% | 316.58 | 348.44 |
| 83 | same_month | recompose_conditional | pooled | 45.459 | 89.01% | 80.80% | 263.56 | 342.31 |
| 83 | same_month | recompose_pooled | max_region | 45.731 | 93.91% | 86.55% | 323.96 | 361.61 |
| 83 | same_month | recompose_pooled | pooled | 45.731 | 88.91% | 82.13% | 269.41 | 346.15 |
| 83 | same_month | unconditional | max_region | 320.281 | 35.65% | 0.00% | 715.45 | 2492.20 |
| 83 | same_month | unconditional | pooled | 320.281 | 35.03% | 0.00% | 702.03 | 2740.70 |

## 条件重组的逐区pooled覆盖

| 地区 | seed | 预算 | CRPS | 覆盖 | 宽度 |
|---|---:|---|---:|---:|---:|
| LGEE | 41 | asof_30d | 27.791 | 98.92% | 264.63 |
| LGEE | 41 | same_month | 19.869 | 91.38% | 151.29 |
| LGEE | 83 | asof_30d | 28.104 | 89.48% | 181.83 |
| LGEE | 83 | same_month | 20.146 | 92.97% | 167.70 |
| PACE | 41 | asof_30d | 88.658 | 93.39% | 522.89 |
| PACE | 41 | same_month | 82.437 | 81.39% | 398.44 |
| PACE | 83 | asof_30d | 88.744 | 82.83% | 427.45 |
| PACE | 83 | same_month | 82.537 | 80.80% | 396.92 |
| SC | 41 | asof_30d | 45.168 | 92.23% | 302.27 |
| SC | 41 | same_month | 33.884 | 93.70% | 239.72 |
| SC | 83 | asof_30d | 45.250 | 93.18% | 290.73 |
| SC | 83 | same_month | 33.693 | 93.26% | 226.06 |

## 配对月份差异

条件重组减去基线；负数代表CRPS改善。12个日历月作为同步块，三个地区一起重采样1,000次，seed20261028。区间为开发样本的描述性区间，不是独立确认或多重比较校正。

| seed | 预算 | 基线 | 相对CRPS变化 | CRPS差异 | 描述95%范围 |
|---:|---|---|---:|---:|---:|
| 41 | same_month | recompose_pooled | -1.22% | -0.559 | [-0.780, -0.286] |
| 41 | same_month | level_residual | -9.71% | -4.880 | [-5.947, -3.843] |
| 41 | asof_30d | recompose_pooled | -2.24% | -1.234 | [-1.644, -0.895] |
| 41 | asof_30d | level_residual | -5.91% | -3.383 | [-4.464, -2.092] |
| 83 | same_month | recompose_pooled | -0.60% | -0.273 | [-0.472, -0.059] |
| 83 | same_month | level_residual | -8.60% | -4.275 | [-5.243, -3.342] |
| 83 | asof_30d | recompose_pooled | -1.71% | -0.940 | [-1.347, -0.572] |
| 83 | asof_30d | level_residual | -4.23% | -2.384 | [-3.370, -1.156] |

## 信息合同与数值限制

同月分支是事后重建上界，不能当未来预测。asof分支是模拟发布预算：历史月总量、质量筛选及小时参考均由回顾性数据生成；没有实际已发布快照，源数据在训练截止时的实际可得性也未逐条核验。无历史月的2024校准/拟合查询按事先可用性统一排除；2025三个目标均保留12月。因此此轮未验证中国真实部署或真实排放。

指标标签为名义固定EF下的非负生产侧代理CIF，未混用因子盒尾数集合覆盖。缺失燃料源形状平坦填补，是模型假设；144个目标查询预算组合中，两种重组池的假想零发电小时质量均为0，当前收益并非由把未定义零发电小时记为0制造。源供体始终为2024拟合地区，目标与校准地区未进入供体库。

本轮完成了重新接回小时指标的强基线实验，未达到原创方法完成条件。下一步应对PACE的当月形状失配、NWMT的统计滞后风险分别归因，再决定是否有可跨区学习的风险传播机制；不能仅继续调全局区间半径。原始小时覆盖、最差地区、CRPS和宽度联合gate继续保留。

依据：[冻结协议](V151_HOURLY_LAW_PROTOCOL.md)、[主结果](../results/v151-hourly-law-r1/summary.json)、[逐区指标](../results/v151-hourly-law-r1/region_metrics.csv)、[输入统计与年龄](../results/v151-hourly-law-r1/query_inputs.csv)、[供体记录](../results/v151-hourly-law-r1/donor_provenance.csv)、[配对差异](../results/v151-hourly-law-audit-r1/paired_month_crps.csv)、[目标零发电质量审计](../results/v151-hourly-law-audit-r1/hypothetical_zero_output.csv)。

输入已备齐、输出目录不存在的干净副本中执行：

```bash
/opt/homebrew/bin/python3.12 scripts/v151_hourly_law.py
```

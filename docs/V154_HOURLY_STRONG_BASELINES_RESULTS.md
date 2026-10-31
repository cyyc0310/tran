# V154：小时重组优于本轮直接条件模型，仍须补锚定残差控制

主实验已确认exit0，12/12训练预算块完成，1,152条目标月份评分、零断言失败。两预算、两源划分、36个2025目标月全部计入；2024 proper源/校准地区严格分离。源抽样360条逐项重建并与存储索引完全一致，整区五折180条角色核对通过。源小时被不同目标fold重复使用，不把总使用次数当独立样本。

## 主要结果与归因边界

V153普通小时重组在概率分布CRPS上优于本轮小时残差模板、分布ExtraTrees、异方差HGB正态。但直接树模型对LGEE严重失效，不能单独依靠这个差距声称方法创新。以第一划分同月预算为例，森林在LGEE的CRPS155.15、覆盖1.04%，而小时重组约19.98、覆盖90.56%。

输入支持审计显示：第一划分proper源2024的能源强度锚最大672.66，LGEE十二个同月输入锚831.12–907.83，全部超出源支持；煤电份额也全部超出源最大54.69%。第二划分源锚上限874.82，仍有5个同月目标超出。这为外推失败提供了明确的可观察输入证据，但没有证明单一特征就是全部原因。

当前直接模型虽有能源锚这个特征，仍须进一步比较“先加已知能源锚，再拟合小时残差”的条件模型。V154小时残差模板是简单锚定对照，不等同于条件残差森林/异方差模型。缺少后者之前不冻结方法、不打开未知地区。

源数据量也须披露：各学习器每源最多4,096小时，同预算模型共用抽样；V153经验形状使用选中源月的全部有效小时。两者合法输入类别相同，但没有逐小时匹配所有源记录；不能把当前差距全部归因于模型结构。后续须补全量源学习或匹配源样本的重组控制。

## 完整宏小时结果

CQR只输出区间/点指标，CRPS为空；CQR点预测取q50，其他分布臂取分布均值，MAE约定有所不同。异方差正态采用1,025点明确离散近似。pooled和max-region半径都完整报告，不能按目标效果选择后声称独立保证。

| seed | 预算 | 方法 | 校准 | CRPS | MAE | 覆盖 | 最差覆盖 | 宽度 | Winkler |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 41 | asof_30d | calendar_residual | max_region | 54.760 | 72.53 | 100.00% | 99.99% | 664.73 | 664.95 |
| 41 | asof_30d | calendar_residual | pooled | 54.760 | 72.53 | 88.56% | 72.76% | 295.25 | 403.36 |
| 41 | asof_30d | conditional_quantile | max_region | — | 185.14 | 95.91% | 89.00% | 900.96 | 932.03 |
| 41 | asof_30d | conditional_quantile | pooled | — | 185.14 | 51.93% | 1.08% | 754.74 | 1757.73 |
| 41 | asof_30d | distribution_forest | max_region | 128.381 | 156.56 | 82.89% | 48.67% | 750.34 | 878.11 |
| 41 | asof_30d | distribution_forest | pooled | 128.381 | 156.56 | 65.32% | 0.21% | 458.24 | 1333.53 |
| 41 | asof_30d | heteroscedastic_normal | max_region | 151.207 | 186.55 | 99.50% | 98.79% | 928.76 | 931.02 |
| 41 | asof_30d | heteroscedastic_normal | pooled | 151.207 | 186.55 | 61.14% | 23.81% | 536.58 | 1320.90 |
| 41 | same_month | calendar_residual | max_region | 47.223 | 60.59 | 94.74% | 84.46% | 350.60 | 392.79 |
| 41 | same_month | calendar_residual | pooled | 47.223 | 60.59 | 79.58% | 51.79% | 191.93 | 434.23 |
| 41 | same_month | conditional_quantile | max_region | — | 109.19 | 85.53% | 72.45% | 814.05 | 900.52 |
| 41 | same_month | conditional_quantile | pooled | — | 109.19 | 59.08% | 0.32% | 759.78 | 1537.86 |
| 41 | same_month | distribution_forest | max_region | 87.788 | 115.33 | 88.99% | 75.58% | 441.73 | 509.16 |
| 41 | same_month | distribution_forest | pooled | 87.788 | 115.33 | 62.77% | 1.04% | 344.56 | 995.04 |
| 41 | same_month | heteroscedastic_normal | max_region | 82.464 | 106.92 | 90.91% | 75.69% | 457.10 | 516.37 |
| 41 | same_month | heteroscedastic_normal | pooled | 82.464 | 106.92 | 64.67% | 11.01% | 335.02 | 734.24 |
| 83 | asof_30d | calendar_residual | max_region | 53.641 | 71.98 | 99.67% | 99.00% | 507.42 | 509.01 |
| 83 | asof_30d | calendar_residual | pooled | 53.641 | 71.98 | 84.75% | 62.85% | 270.85 | 440.90 |
| 83 | asof_30d | conditional_quantile | max_region | — | 125.60 | 90.14% | 74.19% | 887.83 | 1021.39 |
| 83 | asof_30d | conditional_quantile | pooled | — | 125.60 | 60.90% | 49.41% | 727.70 | 1278.49 |
| 83 | asof_30d | distribution_forest | max_region | 72.033 | 99.91 | 98.58% | 95.92% | 654.02 | 659.05 |
| 83 | asof_30d | distribution_forest | pooled | 72.033 | 99.91 | 88.87% | 71.23% | 472.99 | 544.16 |
| 83 | asof_30d | heteroscedastic_normal | max_region | 81.669 | 119.07 | 99.38% | 98.14% | 757.93 | 761.37 |
| 83 | asof_30d | heteroscedastic_normal | pooled | 81.669 | 119.07 | 93.39% | 86.28% | 521.09 | 574.39 |
| 83 | same_month | calendar_residual | max_region | 46.382 | 59.91 | 93.31% | 84.17% | 292.07 | 342.09 |
| 83 | same_month | calendar_residual | pooled | 46.382 | 59.91 | 78.88% | 47.93% | 191.55 | 461.08 |
| 83 | same_month | conditional_quantile | max_region | — | 65.43 | 87.17% | 84.08% | 814.14 | 900.78 |
| 83 | same_month | conditional_quantile | pooled | — | 65.43 | 84.80% | 83.11% | 806.75 | 913.05 |
| 83 | same_month | distribution_forest | max_region | 43.199 | 58.96 | 93.74% | 84.67% | 323.33 | 368.79 |
| 83 | same_month | distribution_forest | pooled | 43.199 | 58.96 | 91.78% | 82.65% | 286.26 | 347.05 |
| 83 | same_month | heteroscedastic_normal | max_region | 49.094 | 65.75 | 99.45% | 98.46% | 451.53 | 454.78 |
| 83 | same_month | heteroscedastic_normal | pooled | 49.094 | 65.75 | 93.76% | 81.93% | 363.85 | 413.35 |

## 对比V153小时重组

下表为小时重组减去V154概率模型的配对CRPS差异，负值有利于小时重组。12个月同步块重采样三个地区，1,000次、seed20261028；描述区间不属于独立确认。覆盖/宽度全表另见比较CSV，部分模板虽然更窄但严重漏覆盖，不能只报单个指标。

| seed | 预算 | 对照 | CRPS差异 | 描述95%范围 |
|---:|---|---|---:|---:|
| 41 | asof_30d | calendar_residual | -7.227 | [-8.863, -5.490] |
| 41 | asof_30d | distribution_forest | -80.847 | [-86.643, -73.960] |
| 41 | asof_30d | heteroscedastic_normal | -103.674 | [-116.767, -89.366] |
| 41 | same_month | calendar_residual | -9.926 | [-11.839, -7.838] |
| 41 | same_month | distribution_forest | -50.491 | [-55.121, -46.409] |
| 41 | same_month | heteroscedastic_normal | -45.167 | [-48.974, -42.054] |
| 83 | asof_30d | calendar_residual | -5.737 | [-7.207, -4.227] |
| 83 | asof_30d | distribution_forest | -24.128 | [-26.572, -21.444] |
| 83 | asof_30d | heteroscedastic_normal | -33.764 | [-42.935, -25.267] |
| 83 | same_month | calendar_residual | -8.793 | [-10.574, -6.836] |
| 83 | same_month | distribution_forest | -5.609 | [-6.685, -4.565] |
| 83 | same_month | heteroscedastic_normal | -11.505 | [-12.806, -10.258] |

## 实现与创新解释

分布森林叶混合属于已有QRF类思路，方差模型采用整区留出残差，均为强基线。数值核验中森林分布均值与点森林最大误差7.11e-15；正态离散在60组积分参数网格的CRPS最大误差0.01542 g/kWh。重复原子的精确合并节省资源，不是统计创新。首次未合并试运行确认终止，最终主实验完整重跑。

默认HGB分位数存在零标签优化退化，另冻结V155严格子梯度控制；不得只用默认失败对照强调胜出。V155也未解决LGEE跨区失效，说明合法子梯度修正不等于解决数据支持问题。详见[V155](V155_STRICT_HISTOGRAM_QUANTILE_RESULTS.md)及[原始论文边界](V154_BASELINE_NOVELTY_BOUNDARY.md)。

本轮保留V153为当前强小时基线，但尚未形成原创机制。下一步优先全量/匹配源数据与锚定条件残差控制，之后再评估是否有值得设计的方法增量。真实发布、中国实证及独立确认要求保持。

依据：[协议](V154_HOURLY_STRONG_BASELINES_PROTOCOL.md)、[主结果](../results/v154-hourly-strong-baselines-r1/summary.json)、[隔离与配对审计](../results/v154-comparison-audit-r1/summary.json)、[全部比较](../results/v154-comparison-audit-r1/hour_vs_strong_models.csv)、[输入支持审计](../results/v154-comparison-audit-r1/input_support_diagnostic.csv)。输入就绪且输出目录不存在的干净副本中执行：

```bash
/opt/homebrew/bin/python3.12 scripts/v154_hourly_strong_baselines.py
/opt/homebrew/bin/python3.12 scripts/v154_compare_strong_baselines.py
```

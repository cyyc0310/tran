# V156：锚定残差强基线收完，重组仍有评分优势但未统一支配

两种源模式均已确认exit0，各864条目标记录，总计1,728条。V157匹配源记录另576条。每模式360条源抽样/全量索引逐项重建、180条整区五折角色核验；共同目标分母完整，所有输入/产物哈希通过。所有地区仍是已查看开发对象。

## 显式锚定的收益很大

同一4,096小时抽样、相同20维输入、相同源角色下，将学习目标从绝对CIF改为CIF减能源锚，并在预测时加锚：第一划分同月森林CRPS87.79→40.05，滞后128.38→54.10；第二划分同月43.20→40.26，滞后72.03→56.19。异方差模型同样明显改善。说明V154直接模型的大量劣势可由标准物理水平残差化修复，不能把此前巨大差距当原创算法收益。

严格分位数对照与V155严格版本比较，均使用相同合法零残差子梯度；不混入默认分位数优化退化的影响。源数据、目标月份和特征相同，只改锚定学习目标。全部差异见归因CSV。

## 与严格匹配源小时重组比较

小时重组相对锚定森林：sampled的同月CRPS约低6.67%/6.49%，滞后低10.40%/13.40%；full同月低7.08%/6.72%，滞后低8.83%/15.32%。因此源记录数量和标量水平锚都不足以消除这一评分差距。

但它没有统一支配所有区间指标。full第一划分滞后，小时重组覆盖95.71%、最差89.99%、宽360.76；锚定森林覆盖92.74%、最差89.87%、宽324.88。两者都过覆盖门槛，重组CRPS更好但区间更宽。sampled第二划分滞后小时重组覆盖89.77%，仍未达90%。不能只报评分优势就宣称完整gate通过。

## 数据量效应

full使用该预算下全部合法源2024小时，sampled复用V154的固定抽样。全量不产生单调改善：第一划分滞后森林改善约2.1%，其它森林设置变化较小或恶化。这个结果只针对当前固定参数管道，不证明来源数据量一般无价值。full保留asof早期无历史统计时的暖启动排除，V157 matched_full使用相同候选源月集合；不与V153未匹配的早期源月混淆。

## 全部宏小时指标

CQR的CRPS为空，点指标为q50；概率模型点指标为分布均值。表内pooled/max-region均完整报告；不得按目标结果选较好校准器后称独立确认。

### 来源模式 sampled

| seed | 预算 | 模型 | 校准 | CRPS | MAE | 覆盖 | 最差覆盖 | 宽度 | Winkler |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 41 | asof_30d | conditional_quantile | max_region | — | 85.57 | 100.00% | 99.99% | 668.78 | 668.99 |
| 41 | asof_30d | conditional_quantile | pooled | — | 85.57 | 92.73% | 89.77% | 343.35 | 382.69 |
| 41 | asof_30d | distribution_forest | max_region | 54.096 | 76.09 | 100.00% | 99.99% | 640.42 | 640.69 |
| 41 | asof_30d | distribution_forest | pooled | 54.096 | 76.09 | 91.42% | 89.88% | 322.30 | 376.52 |
| 41 | asof_30d | heteroscedastic_normal | max_region | 60.533 | 84.63 | 100.00% | 99.99% | 685.01 | 685.24 |
| 41 | asof_30d | heteroscedastic_normal | pooled | 60.533 | 84.63 | 90.47% | 89.43% | 342.52 | 412.89 |
| 41 | same_month | conditional_quantile | max_region | — | 57.00 | 96.78% | 92.71% | 323.94 | 340.82 |
| 41 | same_month | conditional_quantile | pooled | — | 57.00 | 86.94% | 73.96% | 221.34 | 325.22 |
| 41 | same_month | distribution_forest | max_region | 40.046 | 54.40 | 97.47% | 93.16% | 348.25 | 361.58 |
| 41 | same_month | distribution_forest | pooled | 40.046 | 54.40 | 88.38% | 74.43% | 225.93 | 326.32 |
| 41 | same_month | heteroscedastic_normal | max_region | 41.363 | 56.56 | 98.38% | 95.67% | 374.66 | 382.52 |
| 41 | same_month | heteroscedastic_normal | pooled | 41.363 | 56.56 | 87.93% | 73.97% | 226.79 | 324.67 |
| 83 | asof_30d | conditional_quantile | max_region | — | 80.45 | 99.77% | 99.32% | 546.57 | 547.62 |
| 83 | asof_30d | conditional_quantile | pooled | — | 80.45 | 91.24% | 79.62% | 330.36 | 398.29 |
| 83 | asof_30d | distribution_forest | max_region | 56.188 | 77.80 | 99.38% | 98.43% | 538.99 | 541.92 |
| 83 | asof_30d | distribution_forest | pooled | 56.188 | 77.80 | 90.26% | 80.11% | 323.51 | 406.43 |
| 83 | asof_30d | heteroscedastic_normal | max_region | 58.913 | 82.89 | 99.44% | 98.51% | 571.73 | 573.83 |
| 83 | asof_30d | heteroscedastic_normal | pooled | 58.913 | 82.89 | 87.34% | 77.49% | 304.13 | 424.70 |
| 83 | same_month | conditional_quantile | max_region | — | 55.58 | 95.41% | 87.57% | 309.20 | 338.15 |
| 83 | same_month | conditional_quantile | pooled | — | 55.58 | 87.09% | 67.12% | 232.91 | 353.31 |
| 83 | same_month | distribution_forest | max_region | 40.262 | 53.89 | 97.33% | 93.55% | 323.40 | 337.16 |
| 83 | same_month | distribution_forest | pooled | 40.262 | 53.89 | 87.59% | 69.29% | 230.02 | 344.72 |
| 83 | same_month | heteroscedastic_normal | max_region | 40.030 | 54.16 | 96.04% | 92.07% | 297.38 | 317.80 |
| 83 | same_month | heteroscedastic_normal | pooled | 40.030 | 54.16 | 87.69% | 69.94% | 228.95 | 335.99 |

### 来源模式 full

| seed | 预算 | 模型 | 校准 | CRPS | MAE | 覆盖 | 最差覆盖 | 宽度 | Winkler |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 41 | asof_30d | conditional_quantile | max_region | — | 89.77 | 100.00% | 99.99% | 690.21 | 690.43 |
| 41 | asof_30d | conditional_quantile | pooled | — | 89.77 | 92.30% | 86.38% | 347.13 | 398.02 |
| 41 | asof_30d | distribution_forest | max_region | 52.957 | 74.25 | 100.00% | 99.99% | 647.73 | 647.98 |
| 41 | asof_30d | distribution_forest | pooled | 52.957 | 74.25 | 92.74% | 89.87% | 324.88 | 374.66 |
| 41 | asof_30d | heteroscedastic_normal | max_region | 60.624 | 85.65 | 100.00% | 99.99% | 702.08 | 702.30 |
| 41 | asof_30d | heteroscedastic_normal | pooled | 60.624 | 85.65 | 91.39% | 89.27% | 348.86 | 402.44 |
| 41 | same_month | conditional_quantile | max_region | — | 56.55 | 96.25% | 91.03% | 321.71 | 341.98 |
| 41 | same_month | conditional_quantile | pooled | — | 56.55 | 87.58% | 74.44% | 227.13 | 328.78 |
| 41 | same_month | distribution_forest | max_region | 40.141 | 54.68 | 97.90% | 94.28% | 357.39 | 368.41 |
| 41 | same_month | distribution_forest | pooled | 40.141 | 54.68 | 88.81% | 74.90% | 232.19 | 327.65 |
| 41 | same_month | heteroscedastic_normal | max_region | 41.250 | 56.60 | 98.02% | 94.24% | 373.38 | 384.57 |
| 41 | same_month | heteroscedastic_normal | pooled | 41.250 | 56.60 | 85.84% | 72.61% | 212.21 | 319.29 |
| 83 | asof_30d | conditional_quantile | max_region | — | 76.90 | 99.92% | 99.75% | 544.44 | 544.91 |
| 83 | asof_30d | conditional_quantile | pooled | — | 76.90 | 91.46% | 81.24% | 330.34 | 390.47 |
| 83 | asof_30d | distribution_forest | max_region | 57.271 | 78.64 | 99.00% | 97.97% | 531.25 | 536.11 |
| 83 | asof_30d | distribution_forest | pooled | 57.271 | 78.64 | 90.04% | 80.83% | 321.29 | 409.16 |
| 83 | asof_30d | heteroscedastic_normal | max_region | 60.869 | 85.02 | 98.97% | 98.07% | 550.52 | 554.48 |
| 83 | asof_30d | heteroscedastic_normal | pooled | 60.869 | 85.02 | 84.34% | 73.94% | 286.38 | 431.90 |
| 83 | same_month | conditional_quantile | max_region | — | 55.43 | 94.65% | 85.84% | 304.29 | 337.22 |
| 83 | same_month | conditional_quantile | pooled | — | 55.43 | 87.19% | 67.97% | 234.89 | 354.73 |
| 83 | same_month | distribution_forest | max_region | 40.296 | 53.93 | 96.97% | 93.53% | 312.55 | 328.79 |
| 83 | same_month | distribution_forest | pooled | 40.296 | 53.93 | 87.18% | 68.36% | 227.88 | 347.77 |
| 83 | same_month | heteroscedastic_normal | max_region | 40.262 | 54.44 | 95.05% | 88.24% | 289.30 | 320.90 |
| 83 | same_month | heteroscedastic_normal | pooled | 40.262 | 54.44 | 87.41% | 70.11% | 225.82 | 336.09 |

## 后续判断

保留普通小时燃料重组作为强参考，不给它重新命名为创新。它利用来源逐燃料联合小时信息，经目标月组成重新映射成CIF；标量CIF/残差回归尚未完整检验这种来源联合监督的价值。下一步应以同源数据直接学习联合燃料形状/残差分布，并检验物理可行约束和相位迁移的独立作用。这是待验证机会，当前差距还不能归因于某个专门机制。

原始小时CRPS、最差覆盖、效率、真实发布时间、中国实证及独立确认要求均保持。没有打开新地区，V156/V157均无剩余运行进程。

依据：[冻结协议](V156_ANCHORED_HOURLY_RESIDUAL_PROTOCOL.md)、[抽样主结果](../results/v156-anchored-hourly-sampled-r1/summary.json)、[全量主结果](../results/v156-anchored-hourly-full-r1/summary.json)、[抽样比较](../results/v156-anchored-sampled-audit-r1/hour_vs_strong_models.csv)、[全量比较](../results/v156-anchored-full-audit-r1/hour_vs_strong_models.csv)、[锚定/数据量归因](../results/v156-attribution-r1/summary.json)、[严格源小时匹配](V157_MATCHED_SOURCE_HOURS_RESULTS.md)。输出目录不存在的干净副本中运行：

```bash
/opt/homebrew/bin/python3.12 scripts/v156_anchored_hourly_residual.py --source-mode sampled
/opt/homebrew/bin/python3.12 scripts/v156_anchored_hourly_residual.py --source-mode full
/opt/homebrew/bin/python3.12 scripts/v156_compare_anchored_models.py --source-mode sampled
/opt/homebrew/bin/python3.12 scripts/v156_compare_anchored_models.py --source-mode full
```

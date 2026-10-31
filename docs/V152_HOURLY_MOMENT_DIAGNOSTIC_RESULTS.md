# V152：完整月统计下，月内恒定分布已经接近自身CRPS下界

本轮完成1,440条月度诊断评分，逐项复现V151的288条原始重组记录（CRPS、均值、校准上下界及覆盖）。能源/时间均值恒等式和均值误差分解通过。主结果r2给物理边界统计加入1e-8容差，避免把1000附近的舍入尾数计为真实越界；r1为此前数值统计版本，不作为另一次独立实验。

## 找到了一项直接改变路线的约束

设目标月实际小时CIF的经验CDF为H，整月对每个小时给同一个预测CDF为F，则该月平均CRPS满足：

$$\operatorname{CRPS}(F;H)-\operatorname{CRPS}(H;H)=\int (F(z)-H(z))^2\,dz\ge0.$$

这是评分规则的标准恒等式，本轮用20组独立的分段CDF积分核对了前缀和评分。逐月H自评分按地区有效小时汇总、再地区等权后，本批目标下界为43.7403 g/kWh。这个下界仅约束“整月使用同一CDF”的模型和当前已观察的开发样本；不能称为真实未来风险或条件小时预测的不可约噪声。

同月完整统计下，V151条件重组CRPS45.396/45.459，距离该下界只剩3.65%/3.78%；等权重组也仅剩4.82%/4.35%。因此对当前强月内边际基线要求再降5%，在此模型类内已经不可能。进一步调月度分布的均值、尺度或核权重无法突破这个下界。小时/日历条件分布不受同一月内恒定CDF限制，应成为完整月信息分支的下一步。

滞后预算则不同：条件重组CRPS53.872/54.033，距该月内边际下界仍有18.81%/19.05%空间。这允许通过改进未来组成/水平预测取得收益，但不能用事后同月统计冒充发布时点可用信息。

| seed | 预算 | 基线 | 原CRPS | 月内恒定CDF下界 | 最大剩余改善 |
|---:|---|---|---:|---:|---:|
| 41 | asof_30d | recompose_conditional | 53.872 | 43.740 | 18.81% |
| 41 | asof_30d | recompose_pooled | 55.107 | 43.740 | 20.63% |
| 41 | same_month | recompose_conditional | 45.396 | 43.740 | 3.65% |
| 41 | same_month | recompose_pooled | 45.955 | 43.740 | 4.82% |
| 83 | asof_30d | recompose_conditional | 54.033 | 43.740 | 19.05% |
| 83 | asof_30d | recompose_pooled | 54.973 | 43.740 | 20.43% |
| 83 | same_month | recompose_conditional | 45.459 | 43.740 | 3.78% |
| 83 | same_month | recompose_pooled | 45.731 | 43.740 | 4.35% |

## PACE主要缺波动，均值平移帮助很小

完整同月统计、条件重组下，PACE原CRPS82.437/82.537；给真实目标均值后82.333/82.461，改善很小。只给真实标准差则降至80.690/80.733，给真实均值和标准差为80.555/80.630；目标真实经验边际自评分79.941。对应标准差oracle配合冻结源半径，小时覆盖从81.39%/80.80%到90.78%/89.79%，同时区间宽度从398.44/396.92增至456.44/448.75 g/kWh。真实/预测标准差的月中位比约1.21，支持“波动被低估”的定位；但这组oracle仍以更宽区间恢复覆盖，没有给出可部署的缩界方法。

LGEE和SC不同：完整月统计下给真实均值或标准差不一定改善CRPS；例如LGEE第一划分均值oracle把CRPS19.869变为20.901。真实矩不是CRPS最优位置/尺度，不能据均值校正有oracle真值就推断评分必然改善。所有oracle结果都只是机制诊断，不是达到论文gate的方法。

## 发布滞后主要留下可修复的水平空间

滞后预算条件重组的宏CRPS53.872/54.033，真实月时间均值oracle降至45.921/45.758，真实均值+标准差为44.870/44.744，真实经验边际为43.740。相比完整月统计，这里位置误差有较大的诊断改善空间。目标真实均值不可用，这个结果不保证能够从合法历史统计学出来；后续必须用独立源区比较组成动态预测与强持久性基线。

## 已知月度量与未知时间均值的区别

对小时总发电G与CIF I，能源加权强度与时间均值的关系为：

$$\mu_E=E(GI)/E(G),\qquad E(I)=\mu_E-\operatorname{Cov}(G,I)/E(G).$$

完整燃料月总量确定名义EF下的μ_E，仍不确定协方差比κ。因此预测时间均值偏差可以记为历史能源锚变化减去κ迁移误差。κ的真实值本轮从目标小时标签计算，只用于归因，禁止当作预测特征。这一恒等式不构成独立方法创新，也不能绕过源区支持不足。

## 全部宏诊断

以下两个方法、两预算、两划分、五种变换全部报告。覆盖/宽度沿用V151冻结的pooled源半径；oracle原子不裁剪以保留指定矩，最终区间裁[0,1000]。真实边际oracle也使用同一源半径，因此其覆盖不等同于未经校准的90%经验分位覆盖。超出物理范围的原子质量在逐区CSV中保存。

| seed | 预算 | 方法 | 变换 | CRPS | 覆盖 | 宽度 |
|---:|---|---|---|---:|---:|---:|
| 41 | asof_30d | recompose_conditional | oracle_affine | 44.870 | 96.74% | 331.45 |
| 41 | asof_30d | recompose_conditional | oracle_location | 45.921 | 97.13% | 361.24 |
| 41 | asof_30d | recompose_conditional | oracle_marginal | 43.740 | 98.78% | 348.37 |
| 41 | asof_30d | recompose_conditional | oracle_scale | 55.419 | 89.16% | 331.45 |
| 41 | asof_30d | recompose_conditional | raw | 53.872 | 94.85% | 363.26 |
| 41 | asof_30d | recompose_pooled | oracle_affine | 46.158 | 96.80% | 337.75 |
| 41 | asof_30d | recompose_pooled | oracle_location | 47.004 | 98.00% | 395.50 |
| 41 | asof_30d | recompose_pooled | oracle_marginal | 43.740 | 99.14% | 354.56 |
| 41 | asof_30d | recompose_pooled | oracle_scale | 56.844 | 88.68% | 337.75 |
| 41 | asof_30d | recompose_pooled | raw | 55.107 | 95.29% | 397.10 |
| 41 | same_month | recompose_conditional | oracle_affine | 44.817 | 83.26% | 232.61 |
| 41 | same_month | recompose_conditional | oracle_location | 45.733 | 89.46% | 263.15 |
| 41 | same_month | recompose_conditional | oracle_marginal | 43.740 | 90.23% | 254.78 |
| 41 | same_month | recompose_conditional | oracle_scale | 45.305 | 80.63% | 232.61 |
| 41 | same_month | recompose_conditional | raw | 45.396 | 88.83% | 263.15 |
| 41 | same_month | recompose_pooled | oracle_affine | 46.072 | 80.10% | 235.58 |
| 41 | same_month | recompose_pooled | oracle_location | 46.921 | 90.69% | 298.01 |
| 41 | same_month | recompose_pooled | oracle_marginal | 43.740 | 90.35% | 255.75 |
| 41 | same_month | recompose_pooled | oracle_scale | 46.880 | 74.22% | 235.58 |
| 41 | same_month | recompose_pooled | raw | 45.955 | 89.38% | 298.01 |
| 83 | asof_30d | recompose_conditional | oracle_affine | 44.744 | 91.76% | 274.56 |
| 83 | asof_30d | recompose_conditional | oracle_location | 45.758 | 92.98% | 300.43 |
| 83 | asof_30d | recompose_conditional | oracle_marginal | 43.740 | 96.10% | 294.50 |
| 83 | asof_30d | recompose_conditional | oracle_scale | 55.053 | 80.61% | 274.56 |
| 83 | asof_30d | recompose_conditional | raw | 54.033 | 88.50% | 300.01 |
| 83 | asof_30d | recompose_pooled | oracle_affine | 45.771 | 90.36% | 264.55 |
| 83 | asof_30d | recompose_pooled | oracle_location | 46.528 | 92.96% | 302.96 |
| 83 | asof_30d | recompose_pooled | oracle_marginal | 43.740 | 95.90% | 292.15 |
| 83 | asof_30d | recompose_pooled | oracle_scale | 56.203 | 78.58% | 264.55 |
| 83 | asof_30d | recompose_pooled | raw | 54.973 | 88.10% | 302.96 |
| 83 | same_month | recompose_conditional | oracle_affine | 44.630 | 85.30% | 237.09 |
| 83 | same_month | recompose_conditional | oracle_location | 45.581 | 89.38% | 263.47 |
| 83 | same_month | recompose_conditional | oracle_marginal | 43.740 | 90.33% | 255.13 |
| 83 | same_month | recompose_conditional | oracle_scale | 44.843 | 83.96% | 237.09 |
| 83 | same_month | recompose_conditional | raw | 45.459 | 89.01% | 263.56 |
| 83 | same_month | recompose_pooled | oracle_affine | 45.660 | 81.07% | 227.98 |
| 83 | same_month | recompose_pooled | oracle_location | 46.348 | 89.84% | 269.41 |
| 83 | same_month | recompose_pooled | oracle_marginal | 43.740 | 90.34% | 255.33 |
| 83 | same_month | recompose_pooled | oracle_scale | 46.192 | 76.96% | 227.98 |
| 83 | same_month | recompose_pooled | raw | 45.731 | 88.91% | 269.41 |

## 路线调整

完整月分支停止以整月同一CDF的微调作为主创新，转向合法小时/日历条件化；滞后分支则先检查未来组成/水平变化的源区可学性。两者都必须正面对比已有小时模板、条件回归与燃料重组，不能将增加日历特征直接当原创。仍未满足真实发布数据、中国实证、原创机制与独立确认要求；没有打开新地区。

依据：[冻结协议](V152_HOURLY_MOMENT_DIAGNOSTIC_PROTOCOL.md)、[主结果r2](../results/v152-hourly-moment-diagnostic-r2/summary.json)、[逐月矩分解](../results/v152-hourly-moment-diagnostic-r2/moment_decomposition.csv)、[全部逐区诊断](../results/v152-hourly-moment-diagnostic-r2/region_diagnostics.csv)、[模型类下界审计](../results/v152-hourly-moment-audit-r1/summary.json)。输入齐备、输出目录不存在的干净副本中运行：

```bash
/opt/homebrew/bin/python3.12 scripts/v152_hourly_moment_diagnostic.py
```

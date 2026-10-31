# V149：稀少失效的发生与幅度控制

本轮完成4,224条目标评分、96个校准半径，独立实现从保存的源区分数、尺度及曲线重建全部区间，零差异，输入/输出哈希均一致。PACE18、LGEE24、SC24月，两源区划分、两alpha、两阈值、八种方法；所有目标均已查看。这是月度超阈小时数EF盒端点包含率，不是小时CIF预测覆盖。

四种普通学习器在水平/纵向风险曲线上公平比较：全样本对数Ridge、发生概率乘正误差幅度、仅正误差幅度、修正零残差梯度的q90尺度。正样本幅度模型只有17–33个拟合月，且仅来自5–8个源地区；所有超参预注册，没有按目标效果选择。

## 主要证据

只学正误差幅度的水平曲线在LGEE/600两轮均100%覆盖，均宽115.46/76.54h；V146同联合评分水平族为240.75/200.46h。V148规范顺序的严格tie残差CQR为174.50/120.67h，覆盖也均100%，对应缩窄33.8%/36.6%；但V148是逐阈值校准，不能把不同校准任务当作纯机制对照。

这项改善没有扩展成统一优势：同一水平幅度方法SC/600宽222.96/476.75h、SC/900宽330.54/628.88h，均100%覆盖但多数明显宽于普通残差CQR。PACE/600为100%/94.44%覆盖、659.72/420.94h，第一划分输给简单锚与强重组；PACE/900均100%但75.11/19.28h，第一划分仍宽于普通重组28.39h。LGEE/900该法均100%覆盖但716.25/715.88h，接近物理全宽，不构成高阈值效率突破。

纵向正幅度法在SC的91.67%目标阈值行触发完整物理界回退；这是全宽结果，不能称解决调度迁移。水平正幅度法没有这种回退。发生概率乘幅度的hurdle多数退回尺度下限，未产生稳定增量；修正后的q90也没有统一优势。

## 源区发生概率能否学好

下表每个划分只列水平族；两个族初始区间相同，因此失效发生标签和概率指标完全相同，不能算两次独立验证。AP高于发生率，说明排序可能有信号；但全部六个划分Brier均差于拟合发生率常数，尚不支持可靠概率门控。概率诊断仅来自未用于拟合的12个校准源区，没有用目标端点评价或调整概率模型。

| 目标身份折 | seed | 拟合正例/月数 | 校准发生率 | AP | Brier | 常数Brier |
|---|---:|---:|---:|---:|---:|---:|
| PACE | 41 | 19/636 | 0.0488 | 0.1102 | 0.05491 | 0.04670 |
| PACE | 83 | 21/638 | 0.0142 | 0.0242 | 0.02359 | 0.01430 |
| LGEE | 41 | 18/627 | 0.0634 | 0.3201 | 0.08272 | 0.06054 |
| LGEE | 83 | 33/631 | 0.0208 | 0.1910 | 0.02834 | 0.02131 |
| SC | 41 | 17/627 | 0.0386 | 0.1343 | 0.06212 | 0.03729 |
| SC | 83 | 23/629 | 0.0347 | 0.2504 | 0.03502 | 0.03352 |

## 全方法结果

每格为覆盖百分比/平均宽度(h)，分别报告两个源角色划分。α=.20使用同一固定90%惩罚评分口径；完整评分、单端点和联合覆盖在CSV保存。

### alpha=0.1

| 地区 | τ | 方法 | seed41 覆盖/宽 | seed83 覆盖/宽 |
|---|---:|---|---:|---:|
| LGEE | 600 | horizontal_hurdle_mean | 100.00%/240.75 | 100.00%/185.75 |
| LGEE | 600 | horizontal_positive_severity | 100.00%/115.46 | 100.00%/76.54 |
| LGEE | 600 | horizontal_ridge_all | 100.00%/240.75 | 100.00%/200.46 |
| LGEE | 600 | horizontal_strict_q90 | 100.00%/485.96 | 100.00%/697.96 |
| LGEE | 600 | vertical_hurdle_mean | 100.00%/209.75 | 100.00%/154.04 |
| LGEE | 600 | vertical_positive_severity | 100.00%/539.12 | 100.00%/72.50 |
| LGEE | 600 | vertical_ridge_all | 100.00%/209.75 | 100.00%/154.04 |
| LGEE | 600 | vertical_strict_q90 | 100.00%/640.25 | 100.00%/614.08 |
| LGEE | 900 | horizontal_hurdle_mean | 100.00%/720.96 | 100.00%/720.96 |
| LGEE | 900 | horizontal_positive_severity | 100.00%/716.25 | 100.00%/715.88 |
| LGEE | 900 | horizontal_ridge_all | 100.00%/720.96 | 100.00%/720.96 |
| LGEE | 900 | horizontal_strict_q90 | 100.00%/720.96 | 100.00%/720.96 |
| LGEE | 900 | vertical_hurdle_mean | 100.00%/694.58 | 91.67%/690.29 |
| LGEE | 900 | vertical_positive_severity | 100.00%/702.08 | 91.67%/686.04 |
| LGEE | 900 | vertical_ridge_all | 100.00%/694.58 | 91.67%/690.29 |
| LGEE | 900 | vertical_strict_q90 | 100.00%/714.04 | 95.83%/710.21 |
| PACE | 600 | horizontal_hurdle_mean | 100.00%/702.61 | 100.00%/433.61 |
| PACE | 600 | horizontal_positive_severity | 100.00%/659.72 | 94.44%/420.94 |
| PACE | 600 | horizontal_ridge_all | 100.00%/702.61 | 100.00%/433.61 |
| PACE | 600 | horizontal_strict_q90 | 100.00%/702.61 | 100.00%/433.61 |
| PACE | 600 | vertical_hurdle_mean | 100.00%/450.33 | 100.00%/404.89 |
| PACE | 600 | vertical_positive_severity | 100.00%/441.33 | 100.00%/507.11 |
| PACE | 600 | vertical_ridge_all | 100.00%/450.33 | 100.00%/404.89 |
| PACE | 600 | vertical_strict_q90 | 100.00%/434.11 | 100.00%/404.89 |
| PACE | 900 | horizontal_hurdle_mean | 100.00%/135.89 | 100.00%/21.00 |
| PACE | 900 | horizontal_positive_severity | 100.00%/75.11 | 100.00%/19.28 |
| PACE | 900 | horizontal_ridge_all | 100.00%/135.89 | 100.00%/21.00 |
| PACE | 900 | horizontal_strict_q90 | 100.00%/135.89 | 100.00%/21.00 |
| PACE | 900 | vertical_hurdle_mean | 100.00%/145.39 | 100.00%/68.83 |
| PACE | 900 | vertical_positive_severity | 100.00%/131.78 | 100.00%/278.89 |
| PACE | 900 | vertical_ridge_all | 100.00%/145.39 | 100.00%/68.83 |
| PACE | 900 | vertical_strict_q90 | 100.00%/111.67 | 100.00%/68.83 |
| SC | 600 | horizontal_hurdle_mean | 100.00%/259.08 | 100.00%/494.62 |
| SC | 600 | horizontal_positive_severity | 100.00%/222.96 | 100.00%/476.75 |
| SC | 600 | horizontal_ridge_all | 100.00%/259.08 | 100.00%/494.62 |
| SC | 600 | horizontal_strict_q90 | 100.00%/389.88 | 100.00%/361.00 |
| SC | 600 | vertical_hurdle_mean | 100.00%/348.17 | 100.00%/258.17 |
| SC | 600 | vertical_positive_severity | 100.00%/725.00 | 100.00%/715.00 |
| SC | 600 | vertical_ridge_all | 100.00%/261.50 | 100.00%/258.17 |
| SC | 600 | vertical_strict_q90 | 100.00%/269.00 | 100.00%/453.17 |
| SC | 900 | horizontal_hurdle_mean | 100.00%/409.38 | 100.00%/678.67 |
| SC | 900 | horizontal_positive_severity | 100.00%/330.54 | 100.00%/628.88 |
| SC | 900 | horizontal_ridge_all | 100.00%/409.38 | 100.00%/678.67 |
| SC | 900 | horizontal_strict_q90 | 100.00%/552.00 | 100.00%/542.21 |
| SC | 900 | vertical_hurdle_mean | 100.00%/366.08 | 100.00%/358.54 |
| SC | 900 | vertical_positive_severity | 100.00%/725.21 | 100.00%/720.42 |
| SC | 900 | vertical_ridge_all | 100.00%/337.92 | 100.00%/358.54 |
| SC | 900 | vertical_strict_q90 | 100.00%/346.58 | 100.00%/444.50 |

### alpha=0.2

| 地区 | τ | 方法 | seed41 覆盖/宽 | seed83 覆盖/宽 |
|---|---:|---|---:|---:|
| LGEE | 600 | horizontal_hurdle_mean | 100.00%/122.50 | 100.00%/54.96 |
| LGEE | 600 | horizontal_positive_severity | 100.00%/105.96 | 100.00%/53.29 |
| LGEE | 600 | horizontal_ridge_all | 100.00%/122.50 | 100.00%/59.04 |
| LGEE | 600 | horizontal_strict_q90 | 100.00%/161.50 | 100.00%/86.17 |
| LGEE | 600 | vertical_hurdle_mean | 100.00%/148.42 | 100.00%/62.71 |
| LGEE | 600 | vertical_positive_severity | 100.00%/501.88 | 100.00%/59.83 |
| LGEE | 600 | vertical_ridge_all | 100.00%/207.42 | 100.00%/75.79 |
| LGEE | 600 | vertical_strict_q90 | 100.00%/355.88 | 100.00%/226.58 |
| LGEE | 900 | horizontal_hurdle_mean | 100.00%/719.33 | 91.67%/693.25 |
| LGEE | 900 | horizontal_positive_severity | 100.00%/709.21 | 91.67%/689.33 |
| LGEE | 900 | horizontal_ridge_all | 100.00%/719.33 | 91.67%/698.29 |
| LGEE | 900 | horizontal_strict_q90 | 100.00%/720.96 | 100.00%/717.25 |
| LGEE | 900 | vertical_hurdle_mean | 100.00%/691.29 | 91.67%/685.29 |
| LGEE | 900 | vertical_positive_severity | 100.00%/700.25 | 91.67%/685.25 |
| LGEE | 900 | vertical_ridge_all | 100.00%/694.42 | 91.67%/685.96 |
| LGEE | 900 | vertical_strict_q90 | 100.00%/700.71 | 91.67%/692.38 |
| PACE | 600 | horizontal_hurdle_mean | 100.00%/545.00 | 88.89%/390.11 |
| PACE | 600 | horizontal_positive_severity | 100.00%/460.28 | 88.89%/390.11 |
| PACE | 600 | horizontal_ridge_all | 100.00%/545.00 | 88.89%/390.11 |
| PACE | 600 | horizontal_strict_q90 | 100.00%/545.00 | 88.89%/390.11 |
| PACE | 600 | vertical_hurdle_mean | 100.00%/434.11 | 88.89%/385.61 |
| PACE | 600 | vertical_positive_severity | 100.00%/439.72 | 88.89%/383.78 |
| PACE | 600 | vertical_ridge_all | 100.00%/434.11 | 88.89%/385.61 |
| PACE | 600 | vertical_strict_q90 | 100.00%/415.39 | 88.89%/382.83 |
| PACE | 900 | horizontal_hurdle_mean | 100.00%/34.39 | 100.00%/16.28 |
| PACE | 900 | horizontal_positive_severity | 100.00%/19.33 | 100.00%/16.28 |
| PACE | 900 | horizontal_ridge_all | 100.00%/34.39 | 100.00%/16.28 |
| PACE | 900 | horizontal_strict_q90 | 100.00%/34.39 | 100.00%/16.28 |
| PACE | 900 | vertical_hurdle_mean | 100.00%/111.67 | 100.00%/29.72 |
| PACE | 900 | vertical_positive_severity | 100.00%/128.39 | 100.00%/26.11 |
| PACE | 900 | vertical_ridge_all | 100.00%/111.67 | 100.00%/29.72 |
| PACE | 900 | vertical_strict_q90 | 100.00%/72.56 | 100.00%/24.17 |
| SC | 600 | horizontal_hurdle_mean | 100.00%/220.71 | 100.00%/196.54 |
| SC | 600 | horizontal_positive_severity | 100.00%/216.42 | 100.00%/199.88 |
| SC | 600 | horizontal_ridge_all | 100.00%/220.71 | 100.00%/214.71 |
| SC | 600 | horizontal_strict_q90 | 100.00%/269.50 | 100.00%/217.88 |
| SC | 600 | vertical_hurdle_mean | 100.00%/323.38 | 100.00%/214.96 |
| SC | 600 | vertical_positive_severity | 100.00%/720.71 | 100.00%/702.38 |
| SC | 600 | vertical_ridge_all | 100.00%/249.38 | 100.00%/214.96 |
| SC | 600 | vertical_strict_q90 | 100.00%/255.88 | 100.00%/385.54 |
| SC | 900 | horizontal_hurdle_mean | 100.00%/327.46 | 100.00%/334.00 |
| SC | 900 | horizontal_positive_severity | 100.00%/313.08 | 100.00%/343.17 |
| SC | 900 | horizontal_ridge_all | 100.00%/327.46 | 100.00%/365.12 |
| SC | 900 | horizontal_strict_q90 | 100.00%/385.00 | 100.00%/384.12 |
| SC | 900 | vertical_hurdle_mean | 100.00%/350.08 | 100.00%/324.62 |
| SC | 900 | vertical_positive_severity | 100.00%/719.08 | 100.00%/714.46 |
| SC | 900 | vertical_ridge_all | 100.00%/327.83 | 100.00%/324.62 |
| SC | 900 | vertical_strict_q90 | 100.00%/334.62 | 100.00%/392.75 |

## 判断与下一步

正误差幅度的局部收益值得保留为强基线，普通Ridge和hurdle不作为原创。当前证据不支持统一可迁移的风险门控，也未达到小时CIF原始gate。后续优先检验正误差源区的支持范围及发生模型的源区泛化，而不能按LGEE单区改善冻结方法。概率重校准需独立源区角色；不允许直接用当前校准标签调概率，再复用同一标签声称独立conformal保证。

重用的拟合响应为29源教师、校准/目标为30源教师；它们之间的变化是近似。完整同月总量仍是回顾性输入，未来预测的发布时间约束尚待验证。没有打开新地区，也未增加天气/容量预算。

依据：[冻结协议](V149_HURDLE_RISK_PROTOCOL.md)、[结果](../results/v149-hurdle-risk-r1/summary.json)、[全部前沿](../results/v149-hurdle-risk-r1/comparison_frontier.csv)、[概率诊断](../results/v149-hurdle-risk-r1/occurrence_diagnostics.csv)、[独立重建审计](../results/v149-hurdle-risk-audit-r1/summary.json)、[V148修正强基线](V148_QUANTILE_STABILITY_RESULTS.md)。

在输入已备齐、输出目录不存在的干净副本中执行：

```bash
/opt/homebrew/bin/python3.12 scripts/v149_hurdle_risk.py
```

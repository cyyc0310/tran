# V150：LGEE的幅度缩放收益依赖单个源区

主实验完成5,280条月度尾数区间评分；全部特征幅度模型逐端点复现V149的1,056条记录，零断言失败。另完成固定响应教师下的源区删除影响诊断。输入仍仅为完整月度燃料总量及合法源区形状，所有目标均已用于开发，未打开新地区。

## 影响诊断改变了上一轮判断

LGEE/600第一划分原全特征正误差幅度模型为100%覆盖、115.46h；仅从幅度回归样本中去掉AZPS的一个正误差月，保持原响应教师和全部校准地区，最终区间变为720h全物理宽。九维响应几何模型也发生同样变化。这不是重新独立留出AZPS地区的外部验证，而是拟合影响诊断；它已经足以说明该局部缩窄依赖一个拟合样本，不能据此冻结。

逐个去掉正误差源地区时，LGEE/600第一划分全特征宽度范围112.04–720h，第二划分70.96–130.33h，均保持覆盖。几何模型对应112.04–720h和76.21–212.96h。必须检查实际裁剪尺度与最终区间：仅比较未裁剪对数预测的大幅变化会夸大或掩盖实际影响，本轮已重算最终区间。

## 降维与集成没有形成统一增量

十一维组成模型、九维响应几何模型、逐源区删除后的几何预测均值、常数幅度与56维全特征Ridge使用相同预算和校准。几何模型保留LGEE/600约115.46/77.13h的局部收益，并将SC第二划分/600宽度从476.75h降到330.67h；但仍比V148严格tie残差CQR的258.33h宽。PACE第一划分/600几何模型宽707h，未胜强基线。纵向几何在SC第一划分依然接近物理全宽。简单jackknife没有稳定修复这些问题。

独立12校准源区的正误差对数幅度MAE，跨两族六划分描述平均：常数0.651、几何jackknife0.707、几何0.721、组成1.534、全特征1.563。几何/集成各只在12个族×划分中的5个胜过常数；两族的同一失效事件不算独立重复。这是当前拟合器的源区泛化证据，不是月度信息在原则上不可学的证明。MAE针对未裁剪的对数预测，最终区间效率另表评估。

## 全部预注册模型

每格为月份覆盖百分比/平均区间宽度(h)；覆盖对象为假设EF盒下月内超阈小时数两个端点，不是逐小时CIF覆盖。所有18/24个目标月均计入。表中全特征、组成、几何、集成和常数都是标准基线，不主张原创。

### alpha=0.1

| 地区 | τ | 方法 | seed41 覆盖/宽 | seed83 覆盖/宽 |
|---|---:|---|---:|---:|
| LGEE | 600 | horizontal_positive_composition | 100.00%/157.50 | 100.00%/102.79 |
| LGEE | 600 | horizontal_positive_constant | 100.00%/240.75 | 100.00%/200.46 |
| LGEE | 600 | horizontal_positive_full | 100.00%/115.46 | 100.00%/76.54 |
| LGEE | 600 | horizontal_positive_geometry | 100.00%/115.46 | 100.00%/77.12 |
| LGEE | 600 | horizontal_positive_geometry_jackknife | 100.00%/115.46 | 100.00%/76.33 |
| LGEE | 600 | vertical_positive_composition | 100.00%/389.54 | 100.00%/315.12 |
| LGEE | 600 | vertical_positive_constant | 100.00%/209.75 | 100.00%/154.04 |
| LGEE | 600 | vertical_positive_full | 100.00%/539.12 | 100.00%/72.50 |
| LGEE | 600 | vertical_positive_geometry | 100.00%/140.46 | 100.00%/86.25 |
| LGEE | 600 | vertical_positive_geometry_jackknife | 100.00%/353.17 | 100.00%/97.42 |
| LGEE | 900 | horizontal_positive_composition | 100.00%/720.96 | 100.00%/720.96 |
| LGEE | 900 | horizontal_positive_constant | 100.00%/720.96 | 100.00%/720.96 |
| LGEE | 900 | horizontal_positive_full | 100.00%/716.25 | 100.00%/715.88 |
| LGEE | 900 | horizontal_positive_geometry | 100.00%/719.67 | 100.00%/714.38 |
| LGEE | 900 | horizontal_positive_geometry_jackknife | 100.00%/720.71 | 100.00%/714.38 |
| LGEE | 900 | vertical_positive_composition | 100.00%/701.67 | 91.67%/697.96 |
| LGEE | 900 | vertical_positive_constant | 100.00%/694.58 | 91.67%/690.29 |
| LGEE | 900 | vertical_positive_full | 100.00%/702.08 | 91.67%/686.04 |
| LGEE | 900 | vertical_positive_geometry | 95.83%/695.71 | 91.67%/686.88 |
| LGEE | 900 | vertical_positive_geometry_jackknife | 100.00%/706.88 | 91.67%/687.58 |
| PACE | 600 | horizontal_positive_composition | 100.00%/677.78 | 94.44%/418.67 |
| PACE | 600 | horizontal_positive_constant | 100.00%/702.61 | 100.00%/433.61 |
| PACE | 600 | horizontal_positive_full | 100.00%/659.72 | 94.44%/420.94 |
| PACE | 600 | horizontal_positive_geometry | 100.00%/707.00 | 94.44%/420.39 |
| PACE | 600 | horizontal_positive_geometry_jackknife | 100.00%/705.11 | 94.44%/423.78 |
| PACE | 600 | vertical_positive_composition | 100.00%/483.00 | 100.00%/453.44 |
| PACE | 600 | vertical_positive_constant | 100.00%/450.33 | 100.00%/404.89 |
| PACE | 600 | vertical_positive_full | 100.00%/441.33 | 100.00%/507.11 |
| PACE | 600 | vertical_positive_geometry | 100.00%/478.67 | 100.00%/397.28 |
| PACE | 600 | vertical_positive_geometry_jackknife | 100.00%/473.83 | 100.00%/397.83 |
| PACE | 900 | horizontal_positive_composition | 100.00%/103.78 | 100.00%/19.17 |
| PACE | 900 | horizontal_positive_constant | 100.00%/135.89 | 100.00%/21.00 |
| PACE | 900 | horizontal_positive_full | 100.00%/75.11 | 100.00%/19.28 |
| PACE | 900 | horizontal_positive_geometry | 100.00%/126.11 | 100.00%/19.22 |
| PACE | 900 | horizontal_positive_geometry_jackknife | 100.00%/125.89 | 100.00%/19.61 |
| PACE | 900 | vertical_positive_composition | 100.00%/222.56 | 100.00%/169.17 |
| PACE | 900 | vertical_positive_constant | 100.00%/145.39 | 100.00%/68.83 |
| PACE | 900 | vertical_positive_full | 100.00%/131.78 | 100.00%/278.89 |
| PACE | 900 | vertical_positive_geometry | 100.00%/213.67 | 100.00%/53.11 |
| PACE | 900 | vertical_positive_geometry_jackknife | 100.00%/202.17 | 100.00%/54.44 |
| SC | 600 | horizontal_positive_composition | 100.00%/230.00 | 100.00%/545.25 |
| SC | 600 | horizontal_positive_constant | 100.00%/259.08 | 100.00%/494.62 |
| SC | 600 | horizontal_positive_full | 100.00%/222.96 | 100.00%/476.75 |
| SC | 600 | horizontal_positive_geometry | 100.00%/235.50 | 100.00%/330.67 |
| SC | 600 | horizontal_positive_geometry_jackknife | 100.00%/242.62 | 100.00%/360.12 |
| SC | 600 | vertical_positive_composition | 100.00%/725.00 | 100.00%/402.58 |
| SC | 600 | vertical_positive_constant | 100.00%/261.50 | 100.00%/258.17 |
| SC | 600 | vertical_positive_full | 100.00%/725.00 | 100.00%/715.00 |
| SC | 600 | vertical_positive_geometry | 100.00%/722.38 | 100.00%/429.92 |
| SC | 600 | vertical_positive_geometry_jackknife | 100.00%/700.00 | 100.00%/476.12 |
| SC | 900 | horizontal_positive_composition | 100.00%/349.92 | 100.00%/689.62 |
| SC | 900 | horizontal_positive_constant | 100.00%/409.38 | 100.00%/678.67 |
| SC | 900 | horizontal_positive_full | 100.00%/330.54 | 100.00%/628.88 |
| SC | 900 | horizontal_positive_geometry | 100.00%/352.08 | 100.00%/532.46 |
| SC | 900 | horizontal_positive_geometry_jackknife | 100.00%/364.17 | 100.00%/566.54 |
| SC | 900 | vertical_positive_composition | 100.00%/725.21 | 100.00%/461.12 |
| SC | 900 | vertical_positive_constant | 100.00%/337.92 | 100.00%/358.54 |
| SC | 900 | vertical_positive_full | 100.00%/725.21 | 100.00%/720.42 |
| SC | 900 | vertical_positive_geometry | 100.00%/721.75 | 100.00%/507.83 |
| SC | 900 | vertical_positive_geometry_jackknife | 100.00%/696.62 | 100.00%/544.71 |

### alpha=0.2

| 地区 | τ | 方法 | seed41 覆盖/宽 | seed83 覆盖/宽 |
|---|---:|---|---:|---:|
| LGEE | 600 | horizontal_positive_composition | 100.00%/120.29 | 100.00%/54.21 |
| LGEE | 600 | horizontal_positive_constant | 100.00%/122.50 | 100.00%/59.04 |
| LGEE | 600 | horizontal_positive_full | 100.00%/105.96 | 100.00%/53.29 |
| LGEE | 600 | horizontal_positive_geometry | 100.00%/100.25 | 100.00%/53.29 |
| LGEE | 600 | horizontal_positive_geometry_jackknife | 100.00%/98.33 | 100.00%/53.29 |
| LGEE | 600 | vertical_positive_composition | 100.00%/287.62 | 100.00%/116.00 |
| LGEE | 600 | vertical_positive_constant | 100.00%/207.42 | 100.00%/75.79 |
| LGEE | 600 | vertical_positive_full | 100.00%/501.88 | 100.00%/59.83 |
| LGEE | 600 | vertical_positive_geometry | 100.00%/122.71 | 100.00%/71.12 |
| LGEE | 600 | vertical_positive_geometry_jackknife | 100.00%/252.33 | 100.00%/74.79 |
| LGEE | 900 | horizontal_positive_composition | 100.00%/718.88 | 91.67%/692.29 |
| LGEE | 900 | horizontal_positive_constant | 100.00%/719.33 | 91.67%/698.29 |
| LGEE | 900 | horizontal_positive_full | 100.00%/709.21 | 91.67%/689.33 |
| LGEE | 900 | horizontal_positive_geometry | 100.00%/704.75 | 91.67%/689.67 |
| LGEE | 900 | horizontal_positive_geometry_jackknife | 100.00%/702.67 | 91.67%/689.33 |
| LGEE | 900 | vertical_positive_composition | 100.00%/697.08 | 91.67%/687.92 |
| LGEE | 900 | vertical_positive_constant | 100.00%/694.42 | 91.67%/685.96 |
| LGEE | 900 | vertical_positive_full | 100.00%/700.25 | 91.67%/685.25 |
| LGEE | 900 | vertical_positive_geometry | 95.83%/693.04 | 91.67%/686.00 |
| LGEE | 900 | vertical_positive_geometry_jackknife | 100.00%/699.75 | 91.67%/686.12 |
| PACE | 600 | horizontal_positive_composition | 100.00%/492.11 | 88.89%/390.11 |
| PACE | 600 | horizontal_positive_constant | 100.00%/545.00 | 88.89%/390.11 |
| PACE | 600 | horizontal_positive_full | 100.00%/460.28 | 88.89%/390.11 |
| PACE | 600 | horizontal_positive_geometry | 100.00%/500.83 | 88.89%/401.61 |
| PACE | 600 | horizontal_positive_geometry_jackknife | 100.00%/508.44 | 88.89%/401.61 |
| PACE | 600 | vertical_positive_composition | 100.00%/448.83 | 88.89%/382.50 |
| PACE | 600 | vertical_positive_constant | 100.00%/434.11 | 88.89%/385.61 |
| PACE | 600 | vertical_positive_full | 100.00%/439.72 | 88.89%/383.78 |
| PACE | 600 | vertical_positive_geometry | 100.00%/457.11 | 88.89%/385.50 |
| PACE | 600 | vertical_positive_geometry_jackknife | 100.00%/449.17 | 88.89%/385.83 |
| PACE | 900 | horizontal_positive_composition | 100.00%/24.61 | 100.00%/16.28 |
| PACE | 900 | horizontal_positive_constant | 100.00%/34.39 | 100.00%/16.28 |
| PACE | 900 | horizontal_positive_full | 100.00%/19.33 | 100.00%/16.28 |
| PACE | 900 | horizontal_positive_geometry | 100.00%/25.89 | 100.00%/17.22 |
| PACE | 900 | horizontal_positive_geometry_jackknife | 100.00%/27.17 | 100.00%/17.22 |
| PACE | 900 | vertical_positive_composition | 100.00%/148.06 | 100.00%/23.56 |
| PACE | 900 | vertical_positive_constant | 100.00%/111.67 | 100.00%/29.72 |
| PACE | 900 | vertical_positive_full | 100.00%/128.39 | 100.00%/26.11 |
| PACE | 900 | vertical_positive_geometry | 100.00%/166.67 | 100.00%/29.56 |
| PACE | 900 | vertical_positive_geometry_jackknife | 100.00%/148.61 | 100.00%/30.22 |
| SC | 600 | horizontal_positive_composition | 100.00%/224.25 | 100.00%/203.25 |
| SC | 600 | horizontal_positive_constant | 100.00%/220.71 | 100.00%/214.71 |
| SC | 600 | horizontal_positive_full | 100.00%/216.42 | 100.00%/199.88 |
| SC | 600 | horizontal_positive_geometry | 100.00%/218.33 | 100.00%/195.08 |
| SC | 600 | horizontal_positive_geometry_jackknife | 100.00%/218.62 | 100.00%/198.17 |
| SC | 600 | vertical_positive_composition | 100.00%/484.00 | 100.00%/320.33 |
| SC | 600 | vertical_positive_constant | 100.00%/249.38 | 100.00%/214.96 |
| SC | 600 | vertical_positive_full | 100.00%/720.71 | 100.00%/702.38 |
| SC | 600 | vertical_positive_geometry | 100.00%/643.62 | 100.00%/309.21 |
| SC | 600 | vertical_positive_geometry_jackknife | 100.00%/552.46 | 100.00%/328.79 |
| SC | 900 | horizontal_positive_composition | 100.00%/337.00 | 100.00%/346.58 |
| SC | 900 | horizontal_positive_constant | 100.00%/327.46 | 100.00%/365.12 |
| SC | 900 | horizontal_positive_full | 100.00%/313.08 | 100.00%/343.17 |
| SC | 900 | horizontal_positive_geometry | 100.00%/315.79 | 100.00%/330.62 |
| SC | 900 | horizontal_positive_geometry_jackknife | 100.00%/317.75 | 100.00%/337.79 |
| SC | 900 | vertical_positive_composition | 100.00%/498.17 | 100.00%/400.33 |
| SC | 900 | vertical_positive_constant | 100.00%/327.83 | 100.00%/324.62 |
| SC | 900 | vertical_positive_full | 100.00%/719.08 | 100.00%/714.46 |
| SC | 900 | vertical_positive_geometry | 100.00%/639.04 | 100.00%/406.04 |
| SC | 900 | vertical_positive_geometry_jackknife | 100.00%/565.71 | 100.00%/421.71 |

## 路线判断

将条件幅度缩放降为比较基线。继续增加回归特征或给这套缩放命名，目前没有足够依据。下一轮应回到小时分布和月度约束，检验源小时风险支持是否可迁移；若采用支持不足时的回退，必须同时考察覆盖、效率和弃权，不用全宽回退当算法成功。仍须满足原始小时覆盖、最差地区、CRPS和时效性gate，才能讨论冻结及未查看地区确认。

复现：[协议](V150_SEVERITY_SUPPORT_PROTOCOL.md)、[主结果](../results/v150-severity-support-r1/summary.json)、[全部前沿](../results/v150-severity-support-r1/comparison_frontier.csv)、[源区幅度诊断](../results/v150-severity-support-r1/source_severity_diagnostics.csv)、[事后拟合影响审计](../results/v150-support-analysis-r1/summary.json)、[删除源区后的完整前沿](../results/v150-support-analysis-r1/omission_frontier.csv)。两个脚本均拒绝覆盖已有输出，应在输入已就绪的干净副本运行。

```bash
/opt/homebrew/bin/python3.12 scripts/v150_severity_support.py
/opt/homebrew/bin/python3.12 scripts/v150_influence_audit.py
```

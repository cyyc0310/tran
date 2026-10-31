# V310：省际 OD 矩阵能闭合年度空间结构，但不构成小时 CIF 突破

## 结果摘要

审计了 Zenodo v1 记录中 31 省的 2023 年技术发电量、技术 CF、送端—受端转移矩阵、省级 CF 汇总分布及输配电基础设施 CF。矩阵有 116 条非零省际边，总传输量 1,834.23 TWh；生成量合计 9,453.17 TWh。空白转零后，流入和流出总量闭合，数值误差小于 `1e-12 TWh`。正发电技术对应的省级技术因子没有缺失。

比较同记录的省级 CF 均值：

| 计算臂 | MAE (kg CO₂e/kWh) | RMSE | 平均偏差 | Pearson r | Spearman r |
|---|---:|---:|---:|---:|---:|
| 本地技术发电量加权 | 0.0652 | 0.0741 | −0.0546 | 0.9728 | 0.8960 |
| 年度 OD 比例共享流量追踪 | 0.0603 | 0.0616 | −0.0603 | 0.9980 | 0.9919 |
| OD 流量追踪 + 同记录 T&D 因子 | 0.0339 | 0.0362 | −0.0339 | 0.9980 | 0.9931 |

## 可以得出的结论

1. 省际矩阵将空间排序相关从本地技术构成模型的 Spearman 0.896 提高到 0.992，说明对这份数据集的年度省级 CF 汇总而言，已配对的省际流向包含未配对送受总量不具备的信息。
2. 但 MAE 只从 0.0652 降至 0.0603，下降约 7.5%。最明显的一致性改善来自再加上数据集自己的 T&D 生命周期因子：MAE 降至 0.0339。方法上要把“流向识别增量”与“边界项/因子定义补全”分开。
3. 0/31 个流量追踪值、20/31 个“流量追踪 + T&D”值落在该数据集省级 CF 的 Q2.5–Q97.5 范围内。该区间是源数据自己的生命周期模型不确定性分布，均值误差检查并不等于逐点覆盖验证。

## 对 transcif 方向的含义

这是一个实际的**信息结构进展**：中国公开数据中出现了完整省级 OD 矩阵，而非只有分省发送/受入边际；它让 V305 的“最小配对流量观测”路线有了可操作的年度实例。但这份矩阵与其技术因子和目标 CF 来自同一套生命周期研究数据，不能独立确认它代表官方同口径物理流，更不能验证小时 CIF。Zenodo 记录没有说明其省级 CF 的核算侧及与 transcif 目标的精确对齐关系，因此本结果不把目标表直接称为消费侧或发电侧。

因此本轮不宣布算法突破。节点流追踪/比例共享已是电力碳流研究的成熟方法，例如中国电力消费 CF 的时空研究已使用节点流模型；仅把年度矩阵代入比例共享不是原创方法。真正仍有价值的缺口是：在目标地区没有小时 CIF 历史时，如何从可审计的、按发布时间可得的月度技术发电信息与有限 OD/输电信息，构造有校准证据的小时风险预测或锐界，并在独立小时标签上验证覆盖和效率。

## 失败/未验证边界

- 本轮没有用户实际月表；不能把此前“各发电类别都有”的口头描述等同于已核查字段、单位、发布日期、修订史和“发电侧/消费侧”边界。
- 生命周期 CO₂e 与 transcif 的小时发电侧 CO₂ 不是同一估计目标。
- 单年 OD 数据不能给出月度/小时转移结构或政策实时可用性。
- 省内/跨省流量口径、输电损耗处理、间接穿越路径和流矩阵来源尚需论文原文说明；Zenodo record 没关联论文 DOI。
- 没有新地区、未知地区或独立小时标签确认。

## 产物

- 预案：[V310 审计协议](V310_ZENODO_ANNUAL_OD_AUDIT_PROTOCOL.md)
- 复算脚本：`scripts/v310_zenodo_interprovincial_audit.py`
- 逐省结果：`results/v310_zenodo_interprovincial_audit_r1/v310_audit/province_comparison.csv`
- 指标：`results/v310_zenodo_interprovincial_audit_r1/v310_audit/metrics.csv`
- 机器可读摘要：`results/v310_zenodo_interprovincial_audit_r1/v310_audit/summary.json`

## 参考文献

- Xiong, R. et al. *Provincial-level, primary data-focused carbon footprints of electricity in China with uncertainty estimates* [Zenodo dataset, v1](https://zenodo.org/records/20344652), DOI: 10.5281/zenodo.20344652, 2026.
- *Spatiotemporal carbon footprints of electricity production and consumption in China*. Cell Reports Sustainability (2025), [DOI: 10.1016/j.crsus.2025.100466](https://doi.org/10.1016/j.crsus.2025.100466). 该文使用 node-flow 模型生成省级时空 CF，表明节点流本身不构成 V310 创新。
- *Carbon emissions from electricity consumption in interprovincial grids of China: An electricity-extended input-output analysis*. Energy Policy (2026), [DOI: 10.1016/j.enpol.2025.114907](https://doi.org/10.1016/j.enpol.2025.114907)。该文刻画直接、间接跨省供电路径，进一步提示年度/小时目标口径需显式定义。

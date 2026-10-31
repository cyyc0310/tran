# V248：少量本地实测日修复跨区燃料因子失配

日期：2026-10-30。V248协议见[此处](V248_SPARSE_LOCAL_FACTOR_CALIBRATION_PROTOCOL.md)。本结果是月度排放估计的信息价值实验，不是小时风险覆盖结果。

## 结果

以2024同月其他BA实测 CO₂/heat input 率作供体先验，2025目标月的全部本地heat input作规模约束；随机采样目标月`k`个本地日的实测CO₂/heat input比更新燃料因子，50次重复。

| 本地实测日 k | 配对BA-month | 中位APE（均值跨重复） | 平均APE | 总排放WAPE | WAPE重复间SD | 中位绝对强度误差 g/kWh |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 586 | 0.316% | 1.214% | 1.178% | 0 | 1.600 |
| 1 | 586 | 0.0105% | 0.303% | 0.239% | 0.027 | 0.052 |
| 3 | 586 | 0.0064% | 0.168% | 0.129% | 0.013 | 0.033 |
| 7 | 586 | 0.0045% | 0.101% | 0.079% | 0.009 | 0.023 |
| 12 | 586 | 0.0033% | 0.075% | 0.055% | 0.005 | 0.017 |

即目标热输入已知时，仅用**一个本地日**的实测排放/热输入比，已将总排放WAPE从1.18%降至0.24%。三日样本后约0.13%。高发电量主组都改善：煤WAPE 2.31%→0.39%（1日），管道气0.16%→0.14%，单列天然气0.89%→0.19%。油类零日WAPE很高20.56%，一日降至1.92%，但随抽样数变化并非严格单调。

关键压力区也改善：DUK BA的煤/气总预测平均月APE从17.41%（0日）降到4.51%（1日）、2.38%（3日）；WALC从13.54%降到1.84%、1.10%。DUK仍是局部最大误差区，但原来V247的煤强度严重供体偏差能被少量本地校准大幅修复。

## 这是否是论文突破

这是目前最清晰的**最小本地校准价值**证据：跨年供体给排放/热输入先验，目标月heat input锚定尺度，一个当地实测日校准燃料转换因子，能显著消除煤质/机组群造成的地区失配。候选方法可以概括为“外区时间形状/碳因子先验 + 目标区粗热输入 + 少量本地排放—热输入校准”。

但WAPE接近0不是小时预测成功：目标热输入取自整月小时数据聚合；抽样日的CO₂/heat input直接估计了整月的平均排放/热输入率。它只识别了月度水平，完全没有证明未采样小时的CIF形状与尾风险。对中国部署还要验证每个燃料的月热输入/燃料消耗是否可获取，以及本地一日的高频排放观测是否现实。

下一关应把这个本地校准与小时形状分开：固定同样12个本地日预算，用CEMS直接实测小时标签，在单一时区BA上比较generation-only、heat-input anchor及sampled local factor的小时区间覆盖、宽度和Winkler；至少保留DUK/WALC煤电作失败压力样本。只有覆盖维持且区间变窄，才能将候选提升为小时风险预测机制。

复现：[`v248_sparse_local_factor_calibration.py`](../scripts/v248_sparse_local_factor_calibration.py)、[`sampling_curve.csv`](../results/v248-sparse-local-factor-calibration/sampling_curve.csv)、[`fuel_sampling_curve.csv`](../results/v248-sparse-local-factor-calibration/fuel_sampling_curve.csv)、[`ba_sampling_curve.csv`](../results/v248-sparse-local-factor-calibration/ba_sampling_curve.csv)、[`50次逐次结果`](../results/v248-sparse-local-factor-calibration/sampling_repeats.csv)。

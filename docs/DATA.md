# 数据与实验协议

所有时间必须显式带时区，内部转换为 UTC。逐时 `valid_at` 是区间起点；预测起点须整点。地区日历单独指定 IANA 时区，避免夏令时错位。六燃料顺序固定为 `coal, gas, nuclear, hydro, wind, solar`。

## Manifest

可用 `--demo` 产生完整、可读的 CSV/JSON 示例。最小结构如下，`regions` 需包含至少一个独立源域和一个目标域：

```json
{
  "schema_version": 1,
  "track": "issued_public",
  "accounting": "production_direct_co2",
  "cost_currency": "USD",
  "target": "TARGET",
  "horizon": 24,
  "origins": ["2023-01-01T00:00:00Z", "..."],
  "train_until": "2023-07-01T00:00:00Z",
  "validation_until": "2023-10-01T00:00:00Z",
  "regions": [{
    "name": "TARGET", "jurisdiction": "COUNTRY",
    "coverage": ["primitive_grid_id"], "timezone": "Asia/Shanghai",
    "members": [0, 1],
    "config_kind": "published", "forecast_kind": "issued",
    "label_kind": "observed",
    "config_source": "publication URL, vintage and EF/cost methodology",
    "forecast_source": "provider, archive and issue/access latency evidence",
    "releases": "TARGET/releases.csv", "forecasts": "TARGET/forecasts.csv",
    "labels": null
  }]
}
```

路径相对于 manifest。`origins` 应列举完整、有序、不重复的时间，示例中的省略号必须替换。`coverage` 使用共同的最小物理覆盖单元标识；聚合区域须列出其全部组成单元。不能只写名称不同的国家、省份标签来规避重叠排除。成本须统一币种和价格基年，并在来源中记录换算。

`track` 三选一：`issued_public`、`synthetic`、`development_proxy_oracle`。真实公开轨道要求已发布配置及已签发预报，拒绝 synthetic 标签。`label_kind` 为 `observed/reconstructed/synthetic`；重建标签须说明所用 EF 和核算边界，避免把代数一致性误当独立预测能力。当前元数据校验不能自动认证外部来源真实性。

## 三类 CSV

| 文件 | 必需列及单位 |
| --- | --- |
| releases | `period_end, available_at`；每种燃料的 `capacity_{fuel}_mw, mix_{fuel}, ef_{fuel}_g_per_kwh`；`cost_coal_per_mwh, cost_gas_per_mwh, cost_hydro_per_mwh` |
| forecasts | `issued_at, available_at, valid_at, member, temperature_c, wind_speed_ms, shortwave_w_per_m2, demand_forecast_mw` |
| labels（可选于目标） | `valid_at, available_at, cif_g_per_kwh`；六个 `generation_{fuel}_mw` |

容量/发电/负荷为 MW；能源占比之和为 1。风速应为与所用功率曲线一致的轮毂高度代表风速（当前简化曲线按 100 m）；短波为非负的小时平均 W/m²。温度为 °C。不同地区相同编号的成员并不意味着跨区域联合情景；当前模型没有模拟区域间相关性。

统计配置只选择预测起点之前已发布、统计区间已经结束的最新版本。预报选择起点前可获得且覆盖**所有成员和整个预测时域**的最新单一签发批次；不会拼接不同批次或补入实测天气/负荷。修订以 `available_at` 区分，含糊重复记录拒绝运行。

源域标签的区间结束和公开时间都必须不晚于对应训练/验证截止点。缺失、晚到、非有限值和不完整燃料标签会报错，不自动填充。原始数据中的油、生物质或其他燃料不得无声丢弃或重命名；本六燃料版本仅适合该边界完整成立的数据，扩展燃料表时需同时修改调度、标签与 checkpoint 协议。

## 划分与可见性

1. 从源域池删除目标，以及 `coverage` 有交集的所有聚合/子区域；可进一步留整国验证。
2. 源训练窗口结束不晚于 `train_until`；源验证从该点开始且窗口结束不晚于 `validation_until`。目标测试从后者开始。
3. 神经网络、岭回归、可选提升树和残差情景只拟合源域；MAE 主导模式按源域宏平均 MAE 选 checkpoint，CRPS 模式按源验证 CRPS 选模，可显式指定 RMSE。可选学习率搜索使用同一源验证规则，逐候选记录预算及学习曲线。原生点预测与残差构造的概率情景分开评分，概率均值不会暗中替换岭回归/树模型的点输出。
4. 所有模型和目标预测先落盘，之后才可读取目标标签评分；无目标标签时仍可完整训练和推理。

开启机制集合时，内层将相互覆盖的源区域合并为连通组，可进一步按国家合并。每次留出一组，该组既不参与该机制的训练，也不参与 checkpoint 选择。随后只用被留组在源验证时段的预测 CRPS，在预先声明的尺度候选中选一个全局潜在尺度；先平均组内电网，再平均组。该选择分数不是选择后的无偏泛化估计。

最终集合包括全源模型和各留组模型，保持等权及机制身份。配置支持距离仅使用预测前的源公开容量占比和能源占比：到不同源组的最近距离，组内取 95% 分位后取组间最大值作为阈值（数值下限 `1e-6`）。这只能诊断配置外推，不能识别全部分布偏移或保证概率覆盖。`mechanisms.npz` 保留 `N,K,S,H[,F]` 轴；`forecasts.npz` 保存同预算混合路径及消融。条件内与机制间方差是有限情景的代数分解，不是已认证的偶然/认知不确定性分解。

MAE 模式的点输出来自物理层的确定性位置解，训练不读取目标标签；集合点输出为各机制点输出的等权平均。潜在尺度只影响辅助概率情景，不移动已监督的点输出。实验需同时报告 MAE、RMSE 和 CRPS，不能用其中一个指标的进步掩盖另一个的退化。

外层 benchmark 要求评估目标物理覆盖互不相交。所有目标使用固定步骤数、候选和种子，不根据已经产生的外层分数改变设置。输出提供先平均种子后的电网/国家宏平均，不提供统计置信区间。

公开目标历史统计不是逐时遥测，但仍必须证明当时可用。禁止用全年目标 CIF 拟合 EF、从目标标签反推配置、从实测负荷伪造预报，或补造历史预报发布时间。缺乏发布档案的现有数据只能进入明确标注的开发轨道，不能自动升级为真实公开实验。

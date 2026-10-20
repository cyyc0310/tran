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

`selection_protocol=source_group` 时，架构与学习率再通过可见源域内部的留组任务选择：checkpoint 仅使用其余组的时间验证，留组标签用于候选评分，最后重训全部可见源域。机制成员的永久留出组不进入这层搜索。记录中区分候选留域分数与最终重训模型的时间验证分数；至少需要两个独立可见源组。

最终集合包括全源模型和各留组模型，保持等权及机制身份。配置支持距离仅使用预测前的源公开容量占比和能源占比：到不同源组的最近距离，组内取 95% 分位后取组间最大值作为阈值（数值下限 `1e-6`）。这只能诊断配置外推，不能识别全部分布偏移或保证概率覆盖。`mechanisms.npz` 保留 `N,K,S,H[,F]` 轴；`forecasts.npz` 保存同预算混合路径及消融。条件内与机制间方差是有限情景的代数分解，不是已认证的偶然/认知不确定性分解。

MAE 模式默认监督物理层的确定性位置解；显式指定 `median/mean` 时直接监督情景摘要。燃料边际中位数只作为辅助监督统计，不解释为可行的联合调度。集合点输出为机制点输出的等权平均。`physical` 点输出不随潜在尺度变化；`median/mean` 会受尺度、采样数和随机种子影响，尺度校准后必须重报点误差。训练不读取目标标签，MAE、RMSE 与 CRPS 均须报告。

外层 benchmark 要求评估目标物理覆盖互不相交。所有目标使用固定步骤数、候选和种子，不根据已经产生的外层分数改变设置。输出提供先平均种子后的电网/国家宏平均，不提供统计置信区间。

`scripts/compare_benchmarks.py` 可另行做电网/国家与时间块的配对重采样。它核对输入哈希、目标、训练种子和规则采样的预测起点；两方法及各电网共用时间块，保留共同时间冲击。训练种子先平均，不当作独立电网。少数电网、短测试期、非平稳序列及开发选模都会限制区间解释。

## 原始数据重建

已选定美国、英国、澳洲为真实基准范围。`scripts/audit_data.py` 只清点原始证据；2026-10-20 清点的 408 个 CSV 中，166 个 CIF 聚合文件均缺标签来源和显式时区，54 个英国文件的能量字段接近百分比 100，没有文件具备完整签发批次字段。该清点不是来源认证。

旧英国下载器（Git `b5666ef` 的 `scripts/data/download_uk_regions.py`）在 `actual` 缺失时回退到 `forecast`，且输出时移除了 `cif_source`；其能量列由百分比构造。[官方区域产品](https://www.carbonintensity.org.uk/)还包含电力流追踪的消费侧预测，不能当成六燃料发电侧实测标签或 MW 发电监督。

`scripts/prepare_eia.py` 将 [EIA-930 原始 balance 文件](https://www.eia.gov/electricity/930-content/about.html)整理为 staging：将明确 UTC 的小时结束时间减一小时，保留全部原生燃料、缺失值和负净出力，不生成 CIF 或发布时间。不同版本的抽蓄、独立/一体化储能、地热等字段分别保存，并记录逐文件 schema；当前已整理八个美国 BA 的 2022–2024 年数据。staging 元数据明确 `eligible_for_issued_public=false`，不能直接送入六燃料模型。

`scripts/collect_neso.py --start 2022-01-01T00:00Z --end 2025-01-01T00:00Z --output data/staging/neso-national-2022-2024` 从国家接口另存实际估计与预测字段，并保留原始快照、URL、哈希、下载时间。只有两个半小时实际值均存在才形成逐时实际标签；不会用 forecast 回填。下载版本不等于原始发布版本，核算边界、逐燃料 MW 及签发预报仍需补齐。

`scripts/prepare_aemo.py --inputs RAW_DVD_FILES --output NEW_DIRECTORY` 流式整理 AEMO DVD 快照，逐文件验证表版本、完整 footer 记录数及跨数据块主键唯一性，压缩输出和元数据整体提交；失败不留下完整目录。2023 年 28 份快照已整理到 `data/staging/aemo-native-2023`，包括 47,482,096 条 SCADA 记录、473 个 DUID，其中 513,728 条读数为负；读数非空不代表网络边界完整。输入和输出哈希均已复核，原始缓存未修改。

[SCADA 文档](https://visualisations.aemo.com.au/aemo/nemweb/MMSDataModelReport/Electricity/MMS%20Data%20Model%20Report_files/MMS_123.htm)描述的是调度开始时的瞬时 MW。适配器仅将原生区间键按固定 UTC+10 转换，保留 `SETTLEMENTDATE`，不推断五分钟偏移、不生成小时能量或 CIF。注册表保持各快照、原生燃料文本及多机组映射，不按后来的注册快照回填目标配置；报表创建时间和 `LASTCHANGED` 均不写成公开时间。[GENUNITS 的 EF 字段为 CO₂e](https://visualisations.aemo.com.au/aemo/nemweb/MMSDataModelReport/Electricity/MMS%20Data%20Model%20Report_files/MMS_250.htm)，不能直接宣称与当前直接 CO₂ 核算一致。staging 仍为 `eligible_for_issued_public=false`。

`scripts/audit_aemo_registry.py --staging data/staging/aemo-native-2023 --registration-month 202312 --output NEW_REPORT.json` 用显式指定的注册快照，逐月在月中时点检查当月 DUID 清单。保留最新生效日期下各原生版本的机组集合，不擅自解释删除/修订语义；一个 DUID 始终只有一行，避免多机组 join 重复计算 MW。任一候选机组 EF 缺失时不补零或对剩余值取平均。实测每月有 34 个多机组 DUID，26–27 个标为 GENERATOR 的 DUID 缺完整候选 EF。此为事后清单诊断，既非发电量加权缺失率，也非整月逐时映射。[2019 年适用的 CDEII 规程](https://www.aemo.com.au/-/media/files/stakeholder_consultation/consultations/nem-consultations/2019/cdeii/carbon-dioxide-equivalent-intensity-index-procedure-v40final.pdf)还纳入逸散排放，不能通过单位换算得到直接 CO₂ 标签。

天气优先核实完整单次签发档案。[Open-Meteo Single Runs 文档](https://open-meteo.com/en/docs/single-runs-api)明确模型初始化时间不同于公开可用时间；不能直接把 `run` 写成 `available_at`。EIA balance 的 day-ahead demand forecast 也不自带完整签发/修订批次，仍需补原始版本证据。

[NOAA GFS 公开档案](https://registry.opendata.aws/noaa-gfs-bdp-pds/)已实测能读取 2023 年原始预报。`scripts/collect_gfs.py --run 2023-01-01T00:00Z --lead 24 --output NEW_DIRECTORY` 通过带 `If-Match` 的 HTTP Range 只保存 2 m 温度、100 m 风矢量和地表短波 GRIB 消息，同时验证 206 状态、字节范围、对象 ETag、GRIB 长度，保存索引、哈希及 S3 写入时间。初始化、对象写入及当前下载时间分别记录，不自动生成公开时间。

安装 `.[weather]` 后，`scripts/decode_gfs.py --snapshot DIRECTORY --points POINTS.json --output NEW_REPORT.json` 用 ecCodes 复核实际编码的变量、高度、单位、签发批次和平均时段，再提取显式坐标。`hourly_gfs_points` 仅对同一次签发、同一采样坐标的完整相邻时次，按“平均值 × 窗口长度”差分还原小时短波，保留每小时起点的瞬时温度/风；缺锚点或混合批次均报错。负还原量默认报错；显式 `packing_resolution` 策略只将加权编码分辨率阈值内的微小负值投影为零，并保存原值、阈值与次数，更大负值仍拒绝。这是数值处理策略，不是气象误差或概率区间。实测 `f024` 短波是 18–24 h 平均，不能直接标成小时值。

`scripts/collect_gfs_window.py --run 2023-01-01T00:00Z --points POINTS.json --cache CACHE --output NEW_DIRECTORY --solar-negative-policy packing_resolution` 默认构造 06 UTC 起的 24 h 点天气窗口，需 25 个相邻原生时次。网络并发、暂时故障重试和哈希校验后的断点续传均已实现；缓存损坏拒绝覆盖，完整解码后才整体提交输出。对象写入时间须介于初始化与预测起点之间，但仍不自动认证首次公开时间。该真实窗口在美英澳三个示例点已跑通，共 72 点小时，两次微小负短波修正均留痕；示例点不是区域加权天气，也不是性能实验。空间权重、档案可用性协议和匹配配置/需求仍待补齐，staging 维持不可直接进入真实公开基准。

公开目标历史统计不是逐时遥测，但仍必须证明当时可用。禁止用全年目标 CIF 拟合 EF、从目标标签反推配置、从实测负荷伪造预报，或补造历史预报发布时间。缺乏发布档案的现有数据只能进入明确标注的开发轨道，不能自动升级为真实公开实验。

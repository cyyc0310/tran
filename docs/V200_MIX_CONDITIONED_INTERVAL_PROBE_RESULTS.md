# V200：月度燃料组成近邻没有改善可信区间

日期：2026-10-29。V200沿用V199固定的k=15 Bray–Curtis组成近邻与2025中国模型投影小时标签，检验形状信号能否转化为覆盖保持、区间收窄。全部目标标签曾在V194打开；结果是同一模拟世界的来源省留一开发探针，不是独立确认。

## 方法和范围

五情景、31个省、12个月、同情景/跨情景两轨各1,860个省月情景query。所有方法使用事后目标月均oracle锚。目标省及伪目标校准省在其供体档案中的跨情景重复profile都排除。每条预测先写入清单并封存区间哈希，再据目标小时标签评分；这是本轮代码流程隔离，不改变标签此前已可见的事实。

比较Bray–Curtis最近15形状的经验粒子、source-LOO统一残差校准及标准化离散度校准；强参照为按目标月均CIF oracle近邻的k15校准、全30供体统一校准和全30供体经验粒子。统一校准和标准化校准均采用90% higher分位；尺度化因子为LOO源残差除以近邻逐小时样本标准差，scale floor=1e-4。通用尺度化/LOO分位数均为已有方法构件，不作创新主张。

## 宏观结果

| 轨道 | 方法 | 覆盖 | 均宽 | Winkler 90 | 形状MAE |
|---|---|---:|---:|---:|---:|
| 同情景 | 全30经验粒子 | 84.71% | 0.23190 | 0.35865 | 0.06088 |
| 同情景 | 全30 pooled LOO | 89.70% | 0.26871 | 0.36005 | 0.06088 |
| 同情景 | 组成k15 pooled LOO | 89.75% | **0.26031** | 0.34899 | **0.05901** |
| 同情景 | 组成k15尺度化 | 90.03% | 0.26539 | 0.34685 | 0.05901 |
| 同情景 | oracle CIF近邻k15尺度化 | 90.08% | 0.26787 | 0.34640 | 0.06073 |
| 跨情景 | 全30经验粒子 | 82.84% | 0.22993 | 0.37712 | 0.06182 |
| 跨情景 | 全30 pooled LOO | 89.42% | **0.26948** | 0.36554 | 0.06182 |
| 跨情景 | 组成k15 pooled LOO | 89.65% | **0.26505** | **0.35708** | **0.06068** |
| 跨情景 | 组成k15尺度化 | 89.95% | 0.28099 | 0.36488 | 0.06068 |
| 跨情景 | oracle CIF近邻k15尺度化 | 89.99% | 0.28418 | 0.36542 | 0.06251 |

组成近邻尺度区间相对oracle CIF近邻尺度区间没有可区分的收益：Winkler差同/跨情景为+0.00045/−0.00054，省聚类95%区间都跨零；宽度略窄，但配对区间也跨零。相对全30 pooled LOO，同情景Winkler点估计低0.0132但簇区间`[-0.0317, 0.0019]`跨零；跨情景Winkler几乎不变，区间宽0.0115且宽度差95%区间`[0.0053, 0.0182]`表明显著更宽。组成k15 pooled跨情景在点估计上比尺度化更窄、Winkler更低；尺度化并未提供额外优势。

单纯k15粒子分布覆盖仅77.6%–81.8%，说明原始供体粒子不能被当作可靠预测区间。scale/pooled校准使宏覆盖接近90%，但仍没有省级条件保障：组成尺度化的省级最低覆盖同情景57.6%、跨情景59.4%；达到90%的省仅20/31和18/31。五个情景总体覆盖接近，主要失败表现为省际异质性。

## 判定

V199观察到的组成相似度形状收益在点预测上小且跨情景不足；V200没有把它变成更好的区间。在同情景可与oracle月均近邻和全供体pooled校准竞争，但未显示明确区间分数增量；跨情景有的指标更差。不能据总体覆盖约90%忽略部分省份失覆盖。这个分支停止继续搜索k、核宽度或校准分位数。

这既不是算法突破，也不是对用户数据的实测检验：Ember是估算数据，不等于用户原表；CEF小时标签是模拟；锚是事后oracle。V63在其他来源数据上的组成加权负结果与本轮基本一致：低频组成偶尔有形状排序信号，但不足以解决迁移区间风险。Ember字段生成方式和版本边界详见[中国月表审计V60](CHINA_MONTHLY_EMBER_AUDIT_V60.md)。

接下来应回到两个尚未解决的实质问题：一是用户原始月表逐字段和发布时点到底提供何种水平信息；二是公开月度字段能将小时风险的可行集合压缩多少，以及最少还需哪一种本地信息才能认证风险。缺少匹配实测小时真值时，算法有效性的中国实证主张无法验证。这个数据限制不能通过再造相似度/校准器绕过。

## 可复核产物

- 冻结协议：[V200_MIX_CONDITIONED_INTERVAL_PROBE_PROTOCOL.md](V200_MIX_CONDITIONED_INTERVAL_PROBE_PROTOCOL.md)
- Runner：[v200_mix_conditioned_interval_probe.py](../scripts/v200_mix_conditioned_interval_probe.py)
- 宏观与分省输出：[summary_macro.csv](../results/china-projected-cif-v200-mix-interval-r1/summary_macro.csv)、[summary_province.csv](../results/china-projected-cif-v200-mix-interval-r1/summary_province.csv)
- 配对簇bootstrap：[paired_province_bootstrap.csv](../results/china-projected-cif-v200-mix-interval-r1/paired_province_bootstrap.csv)
- 预测封存与逐query清单：[prediction_seal.json](../results/china-projected-cif-v200-mix-interval-r1/prediction_seal.json)、[prediction_manifest.jsonl](../results/china-projected-cif-v200-mix-interval-r1/prediction_manifest.jsonl)

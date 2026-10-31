# V291：GB区域碳强度端点不构成独立历史标签

日期：2026-10-30。仅做官方数据通道预检；没有把区域序列用于拟合、模型选择或评分，也没有打开新目标区的性能结果。

## 核查内容

NESO Carbon Intensity API 文档列出14个DNO区域的半小时碳强度历史查询端点。对region 1分别查询2018-06-01与2024-06-01的一天，API均返回区域ID、DNO名称、半小时区间、`intensity.forecast`/index与区域`generationmix`百分比。两次查询都没有`intensity.actual`或独立实际发电/消费测量字段。

官方项目说明称区域生成、需求按30分钟预测，并通过简化GB网络的交流潮流模型估计各区域消费侧碳强度。区域强度包括区域潮流、损耗和网络阻抗影响；估计对象是消费侧区域因子，和NESO全国燃料结构固定EF代理的生成侧因子不同。

## 对后续验证的限制

- 历史查询端点返回模型预测字段，不能只因时间戳已在过去就称为实际标签。
- 当前网页下载的是持续更新的区域预测CSV，没有历史发布vintage；无法重建过去每个预测起点当时实际发出的预测版本。
- 因此它不能支持“新区域真实小时标签”的盲外测。若未来留存预测档案并等到官方实际值落地，可作为运营预测基线；当前项目没有该可审计配对档案。
- 区域`generationmix`同样是预测量，不应与目标地区实测月度统计或实际燃料边际混称。
- 全国GB历史燃料发电结构仍可用作一个单独的生成侧估算proxy，但V227显示在当前本地小时字段预算下，覆盖提高伴随更宽、Winkler恶化；它不补足14区消费侧标签。

## 来源与收据

- [NESO区域碳强度预测数据页](https://www.neso.energy/data-portal/regional-carbon-intensity-forecast)：明确数据名为Regional Carbon Intensity Forecast，列出下载CSV、方法PDF与NESO Open Data Licence。
- [NESO Carbon Intensity说明及方法](https://carbonintensity.org.uk/)：说明区域30分钟生成/需求预测及用简化GB网络潮流估算消费侧区域强度。
- [官方API文档](https://api.carbonintensity.org.uk/)：列出区域历史查询端点与14个区域ID。
- 只查询region 1的两个单日端点作schema检查；返回结构含`forecast`与`generationmix`，不含`actual`。保存响应仅供字段复核，未使用其中任何强度数值。
- 查询响应与收据已保存：[2018样本](../results/v291-gb-regional-label-preflight/north_scotland_2018-06-01.json)、[2024样本](../results/v291-gb-regional-label-preflight/north_scotland_2024-06-01.json)、[请求与SHA-256](../results/v291-gb-regional-label-preflight/metadata.json)。API日区间各返回49条半小时记录，首尾都含边界时段；如后续构建数据，必须按`from`/`to`去重并明确时间边界。

## 研究结论

GB区域API可用于研究“官方预测产品能否作为比较基线”，但不能解决TransCIF当前缺少独立、可版本化的新区域小时真值的问题。新颖性和泛化结论仍应依赖真正的目标小时观测或明确标注为模型代理的评测轨；不得把另一套模型输出当作真实验证标签。

# 强重组内天气形状开发证据

本包对应[结果报告](../../COARSE_WEATHER_SHAPE_RESULTS.md)和[开发协议](../../COARSE_WEATHER_SHAPE_PROTOCOL.md)。9个跨国配置、两种月报/装机预算、8个固定方法，共624个唯一目标小时。目标国家均已查看；没有打开新国家。

`manifest.json`绑定文件和归档成员SHA。`results/predictions.csv.gz`无损压缩原预测；`members.tar.gz`保留144个原始粒子数组；`source_models.tar.gz`保留9个源头模型和9个同标签轨迹银行。模型为本项目自产joblib，仅用于受信任的本地复现。`snapshots/`保留本轮代码、测试、协议和结果文档。

`cohort_with_labels.csv.gz`沿用前轮2400行评估队列：`region + valid_at + label_row`可绑定预测和代理标签；数组的`row`是该队列的原始行位置。标签在评分后打包，不是模型输入。`label_generation_*`可复算固定因子代理，不能称为实测排放。CRPS只评分原始成员，源CQR仅作用于区间端点。

可以离线复算本轮864组地区指标、32行宏均值及原始CRPS，并检查零增量恒等。国家、地区依次等权；配置再等权的总体不能解释为9个独立国家。完整原生GFS缓存、全源年额外标签银行和原始全小时电量不在本包内，因此这不是完整原始采集、全源年重训或原始公众可用性认证包。原始封存文件保留运行时绝对路径，换机器时需按manifest映射，不应悄改旧封存。

`results/independent_review.json`记录打包前Sol独立复核。`posthoc_oracle_*`故意使用未来真值，只能作误差诊断。`extra_label_reference_crps.*`是额外源年银行参考，不能与同标签主比较混合宣称天气创新。

`climate_probe/`是独立的一周历史天气API语义探针，未加入本轮模型。其原始公众可用性仍为False；不代表完整2016–2020气候参考已采齐。接口字段与历史预报不同，参见[Open-Meteo文档](https://open-meteo.com/en/docs/historical-weather-api)；ERA5早期版与最终版可能有差别，参见[Copernicus数据说明](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)。

本轮结论：半强度天气差分小幅改善点误差和原始CRPS，但覆盖变差，未建立论文方法增量。下一步先检验公开资源参照对日尺度幅度的信息价值，不从已见开发结果挑选强度后再称作确认。

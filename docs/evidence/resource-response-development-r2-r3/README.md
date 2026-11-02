# 日幅度强参考与源配对扩展：中间研究证据

对应[第二轮结果](../../RESOURCE_CONTROLS_RESULTS.md)及[源扩展A/B/C结果](../../SOURCE_RESOURCE_EXPANSION_RESULTS.md)。四次运行均在相同624个唯一目标小时评分，属于已见国家的自适应开发；没有未知国家或中国精度确认。

`manifest.json`绑定文件和归档成员SHA。四个运行目录含全部预测CSV无损gzip、指标、源校准半径、环境、模型/输入/预测seals。`cohort_with_labels.csv.gz`是原2400行队列，`region + valid_at + label_row`可绑定预测标签；细燃料只用于评估固定因子发电侧代理，不代表实测排放，也不进入无标签部署。

本包可独立复算**四组全部点预测和区间评分**。`selected_members.tar.gz`只保留216个粒子文件：每组天气HGB，以及C的无天气HGB/天气RF，覆盖两银行、两预算和9配置；这些主比较的原始CRPS可离线复算。原银行和简单风光的粒子可在[第一轮证据](../resource-amplitudes-r1/README.md)中按manifest映射读取。

其他方法的完整原始CRPS已由独立Sol从本地封存成员重算，记录见`comparison/independent_review.json`；本包没有重复收录所有1584个成员文件。模型SHA和审计保留，大型RF/HGB完整参数仍在各原`results/*/*_heads.joblib`。因此这是精简中间证据，**不是全模型/全原始采集的独立重训发布包**，不能声称仅本包便可重训全部方法或复算所有方法的原始CRPS。

`training_audits.tar.gz`包含36份训练审计。A提供1200小时、真正拟合792小时；按`used_source_days`解读过去月功率凭证，未用Nov/Dec不算拟合。A/B同792小时的无天气模型成员逐值一致；C的35448小时是1477地区日，不是独立国家数。实际支持只有3–9月，US为4–9月。

`source_weather_assets.tar.gz`含9点的原始2022源ERA5响应/receipt、35448行源公开输入队列、同小时792行匹配队列、派生功率曲线气候及各manifest。源队列的wind/solar等列是滞后月度比例，并非隐藏的小时发电量；小时标签从canonical源文件按label_row单独读。观察天气只用于源训练，目标推理仍是原GFS预报。原公众可用性False/None保持，不能说已认证历史原发布版。

`comparison/posthoc_*`使用目标未来真值定位CISO误差，只是事后诊断，不是模型输入、因果归因或新方法成绩。`snapshots/`保留新增实现、测试、协议与报告。原seals保留运行时绝对路径，跨机器按manifest映射，不改旧封存。

结论：源支持扩大后普通天气HGB有本批数值收益，GB/CISO风险与原月报水平问题仍未解决。数据扩展、树模型或普通概率协调不构成原创算法证明；未知国家继续封存。

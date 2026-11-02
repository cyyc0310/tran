# 月报变化量源训练证据

对应[当前状态](../../SOURCE_ANCHOR_TRAINING_STATUS.md)和[源训练协议](../../HOURLY_ANCHOR_BASELINES_PROTOCOL.md)。本包只证明48套源端普通参考的训练、OOF和来源审计完成，**没有新增目标预测、校准或覆盖成绩**，不证明锚定迁移有效。

`results/`含源训练完成、3214项输入封存及98项输出封存；98项为48模型、48 OOF及运行版本/执行计划两份元数据。大型完整模型参数在本地`results/source-anchor-models-r1/models`，不重复收入本包。

`source_oof.tar.gz`保留48份OOF的源row、真实千分比、编码响应、公开锚、编码/解码预测及国家层级权重；`archive_inventory.json`将每个成员绑定原文件与SHA。源row对应[R5源公共队列](../hourly-direct-baselines-r5/source_training_cohort.csv.gz)的行位置。独立比较绝对响应时，使用[R5证据](../hourly-direct-baselines-r5/README.md)中的对应thermal OOF，不混入额外全年标签银行。

`independent_review/`保存独立Sol对48模型/OOF、144留块、R5源标签逐值一致及公式/尺度单位的核验。`source_oof_analysis/`是源内留月份火电占比误差，单位千分比；不使用目标小时标签，却也不是留国家外测，不能当成跨国精度或中国精度。

manifest绑定本包各文件；本包支持源OOF与来源审计，不包含全原始数据或完整重训环境。后续扩展24窗的数据采集和先封存后评分见[独立评价协议](../../EXPANDED_ANCHOR_TRANSFER_PROTOCOL.md)。月报、静态装机代理、源观察天气与签发预报的预算限制继续保留。

# R6a扩展日期、无当期预报证据

对应[R6a结果](../../EXPANDED_ANCHOR_NO_ISSUED_RESULTS.md)和[评分前协议](../../EXPANDED_ANCHOR_TRANSFER_PROTOCOL.md)。所有方法共享历史气候参照，**不使用当期GFS数值**；原月报和静态装机预算分别评分。2400目标小时来自已见US/GB/DE的100地区日，不是未知国家确认或中国精度验证。

`predictions.csv.gz`无损还原331200行预测，`day_metrics.csv.gz`还原13800地区日行；`cohort_with_labels.csv.gz`为全4824行队列，在全部预测封存后生成标签。按country、region、origin、valid_at、label_row可绑定评分，NPZ的`row`或`source_row`均是该队列的位置。目标标签不得用来重训或为该国校准。

`members.tar.gz`含36组配对原始粒子。`components.tar.gz`含72组直接/锚定模型未校准的point、scale、lower/median/upper及proper源校准的相同量；不从截断区间反推原始尺度。`archive_inventory.json`将108个归档成员与原预测seal的SHA对应，可独立复算源校准及全部区间指标。

`results/`保留126条校准、汇总及3389输入/113预测封存；`independent_review/`为独立Sol的完整重算与模型/单位检查。既有源模型与新锚模型仍分别在本地R5及source-anchor路径，[R5证据](../hourly-direct-baselines-r5/README.md)和[新源训练证据](../source-anchor-models-r1/README.md)提供继承关系。包内没有全部大型拟合权重，不冒充全原始重训发布。

普通锚定没有超过固定月报锚或强重组；原银行CISO最差43.33%，异方差90.50%的宏覆盖伴随加宽和局部失败。源H1校准由576扩到2424小时，是额外源校准标签预算，不能把与R5的差值归给算法。完整当期天气阶段尚未完成，所有原始发布认证和代理CIF边界保持。

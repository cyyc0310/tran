# V186输入预检：同一小时预算下的长历史风险特征已准备

2026-10-29，Sol。只有输入准备，没有新小时风险模型、零事件或CIF性能结果。[设计](V186_MONTHLY_HISTORY_HOURLY_RISK_DESIGN.md)仍待补正式模型/评分协议后执行。

六折964个proper source-context、241个test-context检查通过。57,840个日历状态行的20维原始粗输入在short/long容器精确相等；23维历史附加特征及新增41维月度波动统计均有限。新增统计1205行全部由另一套标量均值/成对方差/范围/零比例表达式重建，最大误差8.33e−17。角色、年份、区域prefix和合成发布时点检查通过；12个query在short/long两容器分别修改未发布月量，共24次干预、96个完整特征矩阵比较均不变。

预检只取月量、时间和名单，不加载generation或小时零标签，不增加小时监督。复用的冻结monthly_context在较长prefix中仍拟合既有harmonic Ridge，本轮不用其预测；准确范围是“未拟合新的小时风险模型”，不能称完全没有模型拟合。

有效归档为`results/v186-monthly-risk-feature-preflight-r2`，四控制为direct20 / short43 / long43 / long84。r1的summary曾笼统写models_fitted:false，现已退役保留，旧script/design与原因存于`results/v186-feature-preflight-source-r1`。r2纠正范围字段，未改特征算法；不能用r1 hash证明当前复现。

下一需固定原V171小时IDs、每燃料模型容量及same-budget强控制，先测月度历史对小时零事件风险的价值。更低Brier也不自动意味着CIF覆盖与效率改善；普通特征和分类模型不算原创，原gate及未查看地区隔离保持。

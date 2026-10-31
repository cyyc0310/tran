# V36：固定混合与普通容量范围门控基线

2026-10-26，Sol，首次路由或评分前固定。这是读取V34/V35探索结果后提出的便宜基线筛查，不是确认实验、新机制或支持识别。沿用七个已查看源、原42查询/35评分月和B1信息预算；不训练、不新取库存、不打开小时、旧目标或真正未知地区。

## 输入与规则

固定V35清单`docs/evidence/conditional-diagnostics-v35-verification.json`，SHA `7518b714451d50fc6404b5c784e67ab827c583dd27516d4cc531786935f6e5a9`，并核其完整闭包。容量比例q从已封V34 SizeCap-Ridge的raw training_X及query_feature_SizeCap按名称取coal/gas/oil/other四列；不使用目标细标签计算路由。每outer训练仍排除自身六源、原154例（FPL outer156），按原metadata计source-year。阈值只取该outer训练q，查询年度仍是origin年份减2。

10臂顺序为Global0、Global0.25、Global0.5、Global0.75、Global1、Range4、CoalUpper、OilUpper、CoalOilUpper、AnchorEFUpper。w表示Capacity-OP的权重，另一端是原Size-Ridge。

五个Global臂的w固定。Range4在任一q分量低于训练min−1e−12或高于max+1e−12时w=1，否则0。CoalUpper、OilUpper、CoalOilUpper分别检查对应分量是否超过训练max+1e−12；不检查向下越界。AnchorEFUpper检查容量锚函数400+600q_coal+400q_oil是否高于训练同函数的max+1e−12，不能使用训练真实EF代替。五个门控都只取0/1。

先混合EF粒子e=(1−w)e_Size+w e_Cap，再逐粒子乘原MG_POS，分别计算mean/median摘要。w=0/1的全部原EF/CIF粒子与mean端点逐值保留；不能直接混合中位数。CIF/EF都保留MAE、平方及有符号误差、七源region-RMSE均值，mean与median完整展示，不择优隐藏摘要。

## 解读限制

训练min/max只是少数旧source-year的极值，不是统计支持边界；2025查询库存比训练更新，越界混合了时间变化。每源六起点使用同年度库存，因此路由可能始终相同，本质上是源级选择，不能据此宣称识别月度或小时风险。

EF=400+600r_coal+400r_oil描述实际发电比例r，不描述未知调度映射q→r。gas/other在该EF线性函数中的零方向，不意味着其容量或调度变化无关。CoalOilUpper只是启发式axis-aligned fallback，不宣称由理论推出的目标风险保证。

对于同起点常数w，若容量锚EF在粒子间恒定，混合还把原MG–EF粒子协方差缩为(1−w)倍；收益须与全局混合一起解读。单列ISNE误差抵消与MISO收益，不仅比较宏CIF。CIF下降而EF恶化不支持组成更可信；即使两者均改善也不自动构成创新。

## 执行、输出和判断

五个代码/测试/合同、两份独立code/math预审与环境固定后，exclusive创建新目录/start收据，失败保留，输入首尾哈希核对。只有全部七源成功后写表与completion。复用V35评分支持，不能按容量或成绩删月。零新fit，不创建第二个训练或大型审计流水线；结果独立重算核心公式。

输出route_table420行、training_limits7行（含四维min/max、capEFmax、训练例数/source-year数）、case700、region140、macro20、leave_one_source_out_macro140、label_support42。去掉一个源的表仅作余六源等权敏感性描述，不能据它删源或调整规则。

本轮没有新方法达标判定或未知地区授权。只检查路由是否优于固定混合的EF/CIF权衡、标量及普通全维门控是否同样好，以及收益是否集中在已查看源。全部规则均为普通强基线；专门机制要在同预算下有额外增量。CIF区间宽度、覆盖和小时风险没有在本轮评价。

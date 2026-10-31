# V180：预测组成与小时形状分开传入的普通重组对照

2026-10-29，Sol，新增小时评分前固定。已查看V178及V179粗结果，因此属于继续源开发，不是独立确认。V179最佳粗ES来自历史scalar8，与全月Gaussian的增量变体接近；联合MMD未表现优势。本轮先检验粗支持修正能否传入小时CIF，不预认定原创或增量。

沿用29源六折、2024 asof_30d、原每源≤4096小时、241校准和241测试月。不开新地区或2025，不增加源小时。原完整gate不变。

## 固定方法

原conditional/Gaussian两个参考精确保留。新增九臂，统一使用全月原采样世界库、相同原proper π和源内部形状，且fixed半径都取原Gaussian：

1. rebased_persistence：查询最近合法组成S，不含粗不确定性，世界权重为全月Hellinger。
2. rebased_point_mix：原粗mix_model点组成，权重同上。
3. rebased_increment_hellinger：V179加性组成支持，权重同上。
4. rebased_increment_gaussian_frozen：同支持、原V178全月Gaussian权重，作为固定概率的语义对照，不称最优Gaussian。
5. rebased_increment_gaussian：同支持，重新以该支持的EF锚计算Gaussian权重，Ahat/σ及Hellinger先验保持原值。
6–9. rebased_increment_history_scalar8 / delta8 / logratio8 / joint_mmd80：同支持，沿用V178相应已冻结月度森林概率。不重新挑树容量、响应变换或背景混合强度。

合计11方法，每测试月原参考2×2、新臂9×2半径策略×2校准方式，共40条、9,640评分。固定/自身重校准及pooled/max_region同时报告。不按结果扫描混合强度、地区规则或改覆盖目标。

## 支持和形状

粗世界 `v_d=project_simplex(S_query_past+S_source_current−S_source_past)` 完全沿用V179数值规则。源世界d的燃料内部形状 `q_dtj=g_dtj/(M_source_current_j/T_source_current)`，仅使用原采样小时及合法源月监督；源当前月量为0的燃料以q=1占位，明确是未知形状的平坦假设。新小时原子是 `fuel=v_d*q_dt`，CIF为其EF比率。

同一世界的粗增量和形状保持配对。这不是精确独立控制，也不能单独识别这种耦合的价值。源月量归一不等于原抽样时钟law的均值恰好为1；不声称实现了查询真实T小时的精确月量协调。小时总量为0的原子显式排除并按时钟条件化，保存被排除质量；这种条件化可能改变世界概率和粗一致性，必须报告。不得给全部q加正底以隐瞒零概率。

新增法保存每query每法的source profiles、粗支持、概率、每时钟排除质量及未观测形状能源份额。该份额只描述平坦占位暴露，不能当作误差因果贡献或真实机组容量缺失。

## 检查与判断

预检不评分：源采样身份/预算/角色，零增量保留查询组成，欧氏投影规则；原始M和g同倍放大q/CIF不变；查询未发布M及小时raw不可读检查。正式先构造所有law再读query标签；964原参考必须复现V178。全量审计重建源q、世界支持/概率、物理域、排除质量、校准、评分及汇总，明确共享数值辅助函数的范围。

与相同输入的持平、点组成、重算Gaussian及原Gaussian比较，也与标量/MMD/异方差/PIT强参考比较。即使新法优于V178世界森林，也不自动构成突破。普通加性变化、simplex投影、燃料重组及[概率协调](https://proceedings.mlr.press/v244/zambon24a.html)已有研究，不称论文原创；只有小时覆盖和效率同时有明确收益，才进一步隔离机制并研究专门创新。

固定EF非负生产侧proxy、共享2024源开发、合成发布日、短历史及canonical有效小时月量限制保留，中国实际月报及发布版本未核验。

# V184：更长月度历史与更多粗监督的同预算价值实验

2026-10-29，Sol，任何新增held组成评分前冻结。V181完整独立审计已结束，未达原小时gate；V183粗历史及V184有限smoke已通过。沿用[V184设计](V184_COARSE_HISTORY_VALUE_DESIGN.md)，本轮全部是普通强基线/信息价值实验，不预认定原创。

short / long_fixed / long_all三预算、九种分布方法及公共点持平，共28方法。沿用29源六折、2024相同241个test月、无本地小时CIF输入；不读小时数值、不增加原小时训练预算，不打开2025或未查看地区。long_all仅增加粗信息，所有九基线同预算。共享2024元训练合同保留，不称全框架前向时间确认。

方法参数以已审查的`monthly_history_value_models.py`冻结：持平/严格上一年同月/12月Ridge(alpha12)趋势＋proper联合回测残差；base13标准化Ridge(alpha1)＋proper五地区OOF残差；base13标量ET点预测＋同五OOF RMS Gaussian；四个ET经验叶分布scalar8/delta8×base13/history36，32树/leaf4/max_features1/seed20261028，.95叶+.05 Hellinger(bw.3)。五OOF fold为proper regions按SHA256('v184/coarse/'+region)排序后index%5，scaler及样本权重仅在内折拟合。ET拟合n_jobs2、预测1。Δshare/ΔEF和残差的数学零沿用64eps规则；simplex投影沿用V179。严格上一年缺失即持平，不用两年前同月补。

训练名单从冻结context库存读取并重新核对proper/year/prefix及本预算资格；long_all只取2022–2024合法月，不读2025监督。每query先完成三个预算的27个law及公共点，再读取实际当前组成。保存全部coarse support/概率为打包npz及索引，源feature/prefix/残差和拟合指纹另存。不导出小时原子。

## 评分和数值控制

6,748粗评分、53,984燃料评分。沿用V178的ES、同EF组投影ES、边际CRPS、月零Brier、EF锚CRPS及严格90%覆盖/宽度。补充预先固定的64eps相对容差覆盖诊断（锚和每燃料），不替换严格指标或原小时gate，不依结果改容差。月份等权再地区等权，all29/nonconstant24；公共点持平的旧六个粗指标和逐燃料六个指标必须复现V178的241/1928条记录。

全六折预检不评分：源角色/prefix、数据-only接口、三预算拟合元数据、首尾test query的完整law未来M干预（12query×3budget=36组、324方法law）；short/long_fixed四个base法在预检query精确相等。正式全部241query再做相同负控制（964方法law）。另外核对短预算seasonal等于持平、两原子toy的精确loss、strict/tolerant仅作数值诊断。

逐fold原子提交，manifest/hash不变才允许resume；预检与正式代码/数据完全一致。全部模型重新拟合，保存和重建所有源矩阵/残差/指纹、叶概率、OOF控制及预测law/评分/汇总。独立loss表达式及源内标准化/OOF训练范围需明确；共享的scikit-learn拟合、投影/分位数不可写成完全独立算法验证。

## 判定

同一方法比较short→long_fixed→long_all，隔离输入历史和监督池；long_all内与持平、季节、线性和Gaussian比较，不只报对短预算弱参考胜出。可列原V178/V179粗参考，明确它们信息预算不同。bootstrap为反复查看的地区配对描述，不是独立显著性确认。

本轮不定义新的小时达标条件、不选择未查看地区。若无稳定粗增量，停止向小时扩大；若有，仍须通过普通小时传播及全部同预算强控制，满足原覆盖/效率gate后才冻结机制。粗层低ES或更高覆盖不能直接称CIF突破。

canonical有效小时月量、负值proxy、共享2024、合成发布日和中国实际月表未核验等限制保留。此次是控制信息价值，非真实issued-public月报实验。

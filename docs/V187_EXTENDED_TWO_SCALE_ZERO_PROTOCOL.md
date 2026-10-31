# V187：同长历史预算的月状态与正月内零风险强控制

2026-10-29，Sol，任何新增性能评分前冻结。V186已查看，本轮是同源开发的可证伪解释检验，不是确认或原创方法。待检验解释：长历史部分收益可能来自月燃料状态持续性，而非正月内调度风险的普遍可迁移信息；不能预认定该解释成立。

## 数据与固定方法

沿用29源2024六折、原proper每源≤4096小时及全部241 test/241 calibration查询。新增661月仅为合法粗输入/粗监督，不增加小时标签，不打开2025或未查看地区。保留V186全部10个hourly概率法；它们冻结读取，不声称本轮重拟合。原短历史world/hurdle仍是信息受限历史参照；long43/84直接森林已获得相同extended输入。

新月状态m=P(M_current,f=0|已发布prefix)三头：base13、history36、history77（36+V186已固定41波动统计）。三者使用V184各折同一long_all proper 2022–2024合法粗标签名单（585–628月），每燃料32树、leaf4、max_features1、seed20261028、地区平衡权重。只用proper当前月量作监督，query输入严格release≤origin。

正月内风险r=P(hour_zero|M_current,f>0,X)两头：long43/long84，按源当前月燃料量为正过滤**同一原2024源小时**，每燃料32树/leaf64、相同seed/地区平衡。过滤不增加小时，V186保存的X/raw索引/target必须逐项精确复现。普通positive-month clock用同一过滤集的48状态加权均值，必须复现V171保存clock；空状态用原全部正月源小时回退。另保留V171短历史conditional43作为条件任务对照。

每个新月头分别与clock、long_cond43、long_cond84组合h=m+(1−m)r，固定完整3×3=9法，不选燃料或地区；另加最近合法月状态持平×long_cond43/84两法。11新增+10旧=21个部署hourly方法，每test月21×8=168，共40488评分。新的24个月森林+16个条件小时森林=40/折、240/六折；不把monthly_context内部既有harmonic Ridge计算冒称完全无拟合，其预测不用于本实验。

## 三类任务与分母

1. 部署无条件小时任务：21方法×241×8=40488记录，包含真实月为零和为正的所有test月；先构造所有概率，再读当前小时零标签。按有效小时加权后地区等权，all29和预测时可测input_fuel_positive分层全部报告。
2. 月事件任务：V171原5方法（world仅源月零proxy，不能称生成器全部整月零事件概率）+3新增月头=8方法，共15424记录，月份事件损失沿V171同hours权重以便精确复现，不冒称独立月数增加。
3. 明确条件任务：仅实际M_current,f>0的1136个query/fuel，clock、V171短conditional43、新long_cond43/84，共4544记录。**current-positive为评估该条件定义所需的事后标签，不可作为部署筛选或无条件校准断言。**真正部署用上述完整hurdle概率，不把当前月状态当已知。

三类任务都保存cal/test概率，只有test评分。V186旧19280 hourly评分、V171旧9640月事件与2272条件评分必须复现≤1e−12；旧保存概率逐query复制校验，clock精确复现。整体Brier由各真实状态的贡献组成，但源组异质/月份选择可能改变，不能因条件子任务变化唯一归因全部部署收益。

## 预检、审计与比较

六折无held评分预检重拟合240森林，首尾12 query同时扰动所有未发布query M/shares并将raw置None，hourly/月事件/条件三类概率须逐位不变；只在合法proper源构造阶段读取源小时。正式拟合树指纹与预检完全一致，保存源小时/粗标签特征和目标、过滤mask、prefix/provenance、全部概率；完整hash开始/结束校验，同manifest原子fold提交/resume。预检调用正式prefix序列化分支防归档错误。

完整审计重拟合240森林，重建全部482查询概率、源矩阵/索引/过滤/prefix/树指纹、三任务评分及汇总。Brier以独立p²−2py+y展开核验；scikit-learn树生长及冻结feature helper共享，不能称第二套独立预测算法。检查条件任务真实正月的身份及分母，不把该任务当新增标签或验证集。

预定描述比较：11新增hourly各自对long43、long84、short43、world_gaussian、V171持平×条件森林、原history hurdle；另固定同新月头cond43/84对clock、cond84对cond43，以及同条件头history/base、volatility/base、volatility/history和新月头对持平。月任务比较三新头互比及旧持平/base/history；条件任务比较long43/84对短conditional及clock、并long84对long43。全部8燃料、all29/input-positive报告，地区配对bootstrap10000次seed187，只是反复源开发描述，不作多重选择修正或独立确认。

## 决策与边界

只检验总体及条件任务信息价值，不将旧短历史持平条件法当成长预算优化控制。若长预算hurdle/直接法没有稳定整体增量，且正月内条件风险不改善，停止继续堆分类特征，回查信息合同与可识别风险边界；相反结果也只说明普通强控制有效，不能自动推成新机制或CIF效率收益。没有新增CIF预测，原CRPS/覆盖/宽度/Winkler gate不变，未查看地区不晋级。

固定EF非负生产proxy、共享2024元训练、合成发布日、回顾性月量、反复源开发和中国实际月表/版本未核验限制全部保留。普通hurdle、森林、月状态或波动统计不是原创。

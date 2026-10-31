# V31：冻结源上下文模型，检验无当月oracle的月初粗预报

2026-10-26，Sol，运行前合同。V30找到过去绝对规模的强条件关联，但十四原信号均false、当前g为oracle、规模敏感，尚无原创机制或可信小时结论。本轮不改V30记录，不调新参数；检验已冻结方法在前瞻信息预算下的粗点预测。

## 来源与冻结

父清单`docs/evidence/coarse-context-v30-verification.json` SHA `b3590c85d46d99510403432a24c658f1df85137831a3c46005054f0df994cf4a`及其完整闭包固定。七源仍BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS，不开旧目标或真正未查看地区。每outer的V30 sealed JSON提供其余六源L24训练X/y/权重/scaler／参数；模型仅保留G5、Trend、Size各Ridge／ET，共六臂。所有训练数据、特征顺序与scaler原样冻结，不使用2025隐藏标签重选、调参或增训练。

生产从已封metadata重fit同一六head，七区42次重fit（21Ridge/21ET），属于相同模型再现，不是新增2025学习。训练keys／regions须为六个非outer源、标签start在2022Jan–2024Jun、end/available≤部署2025Jan1；六head同队列／y／权重。Ridge训练统计只作原scaler一致性检查，fit/predict仍使用父scaler；系数／截距须重现至1e−10。ET原rawX、128tree/leaf2/seed17023/njobs1，importance须重现至1e−12。不得读取父old query_X来给新预测加oracle。

## 42个固定起点与合法输入

每区2025Jan1至Jun1六个月初，始终预测42个source-origin，不依hourly有效日mask或未来标签有无筛选。当前g只从父链原V25预测NPZ的origins／MG_draws取得该日128五组粒子，不读取其中小时路径、fine_draws或其它预测字段，亦不读取当月真实ZERO／POS／total。原MG law及粒子原样保持，不能合并M8 forecast冒充MG。

每query origin重新按同区metadata选择最近三条自2022Jan起、start<origin、period_end≤origin、available_at≤origin的公开月行，latest-first，再转换四ZERO补POS和过去total_mwh／calendarhours。context可随公开统计合法更新至2025过去月；训练head仍L24，不能错误地把query history也截到2024Jun。当前／未来hidden四项永不为context转换；当前行以后成为合法过去属允许预算。三条history不足则拒绝整个预测attempt，不按不同family换样本。日期必须为2025Jan–Jun合法UTC月初，standalone日期guard先于row数值读取／fit。

V30相同feature顺序构造G5(5)、Trend(30)、Size(31)；季节、过去份额、月龄、发布年龄、log趋势与log规模按各query origin计算。完整数据是修订EIA／30d模拟release代理，非中国类别／真实vintage认证。源区／2025数据已反复研究，不能把此时间检验叫全盲确认。

## 点预测与目标定义

每个g粒子预测raw r4，最后一次simplex投影，内部EF=r·[1000,400,800,400]。每粒子完整月代理CIF=g_POS×EF（原ZERO的EF为0）。EF_point=mean(EF_draws)，CIF_point=mean(g_POS×EF_draws)，**不能用mean(g_POS)×mean(EF_draws)**，也不能先平均g再跑非线性投影。

目标真值为每个完整月八类生成份额·原固定EF8，即月内按生成能量加权的发电侧强度代理；不等于等小时均值、消费侧碳流或实测排放。误差先地区内月份等权均，再七区等权均；不是按不同月份电量加权MAE。主要指标完整月CIF MAE／RMSE，内部EF与G5／POS预报误差为辅助。

128粒子只有原MG预报散布与条件head映射，没有完整隐藏燃料／调度风险或规范校准，因此本轮**不报告概率区间覆盖或可信小时风险**。没有raw hourly CSV数值读取，没有天气／年度细分新预算。

## 标签支持与零POS

全部七预测档案封存、exact14 regular非symlink库存及hash验证后，才读七源2025Jan–Jun当月隐藏八项真值。评分标签截止2025Sep1；须whole-month metadata合法、available_at≤该截止、coverage≥0.9、total_mwh>0。这些字段只用于事后真值质量，不进预测features；future/unreleased label在数值转换前排除。42起点全部写label_support.csv，缺row／未发布／coverage不足／非正总量分别记录，不按误差选择。已选标签数值非法直接失败，不能悄悄换样本。

六臂共享完整月CIF的相同label support。POS=0且总量>0时CIF真值仍为0，必须保留primary；内部EF不定义，记ef_defined=false、其误差NaN，辅助EF只在defined月份聚合并记录计数，不能把未定义EF补0，也不能因此删primary月。

## 表、比较与信号

预测每区6×6×128个r样本（arm×origin×particle）。保存raw/projected r、EF/CIF draws与点、projection_l1、MG_draws、三family query features、context全部provenance及六head再现metadata。

case表每有效source-month×6，region42行，macro6行。主指标cif_abs_error/cif_sq_error/cif_signed_error，辅助ef_abs/sq/signed_error、g5_l1、pos_abs_error和projection_l1。地区RMSE=sqrt(地区MSE)，宏RMSE为七区RMSE平均；internal EF另计有效月，允许明示undefined。label_support始终42行。输出六文件：case/region/macro/matched_contrasts四CSV、source_choice.json、label_support.csv；加freeze/seal/completion为9根文件。

固定9对：每model依Size−Trend、Size−G5、Trend−G5共6个feature比较；另G5/Trend/Size的ET−Ridge共3。每对七区=63 contrasts，全部保留。

对六feature比较，以主要完整月CIF MAE用固定信号规则：宏至少下降5%、至少四区严格改善、任一区不超参考110%（数值容差1e−12）、MISO与ERCO均不恶化，参考宏误差须>1e−12。Size信号须同学习器对Trend与G5两对同时通过。EF只辅助诊断，不拿它改判V30原false；物理目标在本轮结果前固定。未过也完整报告绝对代价／近零参考敏感性，不后验放宽规则。source_choice的hourly_stage_authorized/new_holdout_authorized/mechanism_breakthrough始终false；普通baseline的信号不自动证明原创。

## 执行与独立核验

档案与整数日期统一为UTC纳秒；构造DatetimeIndex后显式转换为ns，不能依赖不同pandas版本的默认精度。微秒DatetimeIndex输入与同一时刻的纳秒输入应产生相同context；整数日期及原MG档案仍严格要求纳秒。正式环境为`../tran-stable/.venv/bin/python`（pandas 2.3.3）；在pandas 3.0.3下另跑合成兼容检查。

五producer文件及三最终预审固定后才prepare。prepare/predict/score均exclusive写，不覆盖失败；全7预测先seal后score。freeze须绑定完整父闭包、每区V30 head JSON及原MG forecast路径，所有实际消费alias有固定hash。

独立actual auditor不导入V31producer，先完成收据／代码／父闭包／环境／9根及14预测绑定再数值；仅读取原MG档的origins/MG_draws、V30训练metadata及raw monthly CSV，重建context/features、42冻结head再现、逐粒子r／mean(product)、label support、42region/6macro/63contrast与六signal，首尾hash不变。保留端点舍入的分阶段核验须明确说明。科学最终审核后才封清单。

本轮仅检验去除oracle后的粗前瞻预算。前瞻效率有效、小时风险、自己的机制增量及冻结后的真正未查看地区确认仍是不同要求；长期目标active。

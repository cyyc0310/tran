# V30：公开月度上下文的跨区粗组成条件诊断

2026-10-26，Sol。运行前合同。V29首次普通异方差锚通过开发门槛，但FPL余量仅0.056pp，MISO收益伴随ERCO损失，尚无原创或稳健迁移结论。本轮检验完整公开月表中未充分利用的历史与总量信息；**仅点预测的粗尺度条件诊断，不是未来小时预测或区间覆盖实验。**

## 数据、时点与oracle条件

父清单`docs/evidence/conditional-law-v29-verification.json` SHA `7512d5e67a21073e42150f7ffb3b379cdd0d03f1061bdcff18348a8c9253b35f`及其完整字节闭包固定。唯一新数值来源仍为既有七源monthly.csv；不解析任何小时CSV、天气、年度细分、旧目标或真正未查看地区。hash既有父文件不视为新数值输入。

七源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS，每outer留整区、训练其余六区。标签月份L24为2022Jan至2024Jun，period_end和available_at均≤固定部署截止2025-01-01；没有季度插补或新增年度。原始表是修订EIA／模拟release代理，不认证中国字段或真实vintage。用户有完整分类月发电量，但火电内煤气油仍未确认分列。

case的origin固定该月period_start。过去context按origin筛同区、start≥2022Jan且<origin、period_end≤origin、available_at≤origin的最近三行，latest-first。每个case分别先筛字符串metadata再转换选中三行数值，不能用部署时全历史构造早期case的过去。当前月total不作输入，但该行在晚些case里可以合法成为过去；毒值测试必须据此构造terminal case。只因不足三行history而skip，已选合法行数值错误不以换更早行掩盖。

当前G5由四个ZERO份额及POS=1−sum(ZERO)得到，**不转换coal/gas/oil/other来构造features**。当前可见POS≤1e−12不进入标签队列，tiny补集舍入按此规则处理；历史零POS允许。过去三行各类可见组成与total_mwh／日历月小时数得到平均MW，须有限正值；不使用observed_hours/coverage等用户未确认字段。重复region-month拒绝，日期／发布顺序／simplex异常拒绝，未来与未发布数值不进入case构造。

**本轮query当前g是该标签月已实现的公开五组组成，明确为oracle-current-group条件。**隐藏r仍不提供给预测器。它用于判断即使给定当前组状态，合法过去context是否有助于逆推火电内部结构；并非月初可得的前瞻预报。若以后接小时阶段，当前g必须换回合法MG forecast粒子积分，不能搬用此oracle，粗尺度改善也不等于小时改善。

## 同队列五组特征与固定学习器

所有特征、两学习器共用完全相同的positive/current与三条context及过去正总量队列；按每次训练重新地区等权。不得拿旧178/180月模型与新较少队列直接归因，也不为每个family另删样本。精确feature顺序：

| family | 内容 | 维度 |
| --- | --- | ---: |
| G5 | 当前POS/核/水/风/光 | 5 |
| Season | G5＋sin/cos(2π(month−1)/12) | 7 |
| History | Season＋最近三行MG5展平＋三条整数月龄＋三条releaseage_days | 28 |
| Trend | History＋log(MW0/MW1)、log(MW1/MW2) | 30 |
| Size | Trend＋log(MW0) | 31 |

月龄为origin月序−context月序；releaseage为(origin−available_at)/day。同源上下文可有日期缺口，月龄真实记录，不把三行伪装连续三月。规模用过去平均MW，单位固定MWh／calendar-hour，不用当前月实际总量、目标地区ID或隐藏燃料特征。

ARMS按family顺序，每family依Ridge、ET排列，共10。Ridge(alpha=1,fit_intercept=True)，X按**训练队列与训练权重**的均值和总体标准差标准化，std≤1e−12置1；query不参与scaler。ET(n_estimators=128,min_samples_leaf=2,max_features=1.0,random_state=17023,n_jobs=1)，使用rawX。每fit样本权重`N/(R*n_region)`，地区总质量相同、区内月等权、均值1。固定模型无新增调参网格。

每outer训练10个head，七区70fit。只在排除outer所有隐藏标签后转换训练八项y，r=y_POS/sum(y_POS)。预测r4最后一次simplex投影；EF_point=r·[1000,400,800,400]。这只是POS内部代理EF，**不是完整地区月CIF、更不是小时CIF**。无风险方差或校准拟合，因此不引入不必要的inner teacher fold。

纯module不得I/O。归档各fit训练／query rawX、y、keys、权重、期间／release、Ridge scaler/系数或ET参数/importance，以及raw/projected ratios、EF、projection_l1、可见context全metadata。outer隐藏数值毒值不得改变cases或该outer预测；可见currentg和合法past量是允许输入。

## 封存、指标与匹配对比

prepare冻结5producer文件＋完整父闭包＋code/python/science预审。predict构造可见cases并完成七个outer档案，exclusive写，不覆盖失败；在全部七档seal并验证exact14 regular非symlink文件后，score才转换七源query隐藏r响应。仍只七已查看源开发，不是新地区确认；不解析任何小时响应。

每case每arm保存ef_abs_error、ef_sq_error、ef_signed_error、ratio_l1、coal_abs_error及projection_l1。先区内case等权均，再七区等权宏均；地区ef_rmse为sqrt(区mean ef_sq_error)，宏ef_rmse为七区RMSE均值，不能混称pooled RMSE。region共70、macro10行；case行=eligiblecases×10，实际cohort数与skip原因完整记录。月相关性不按独立样本量造置信结论。

每model有相邻Season−G5、History−Season、Trend−History、Size−Trend四对，另History/Trend/Size各−G5三对，共14；另每family ET−Ridge五对，总19对×七区=133行。每对指标均保留delta，不因正负删地区。

预定“有粗尺度增量信号”的描述规则：同学习器比较，宏EF MAE至少下降5%、至少四地区严格改善、无地区EF MAE超过参考110%（数值容差1e−12）、MISO与ERCO均不恶化。参考宏误差≤1e−12则不签有意义的相对增量。该规则分别用于14个feature比较；Season效果不冒充历史／绝对量效果，只有History/Trend/Size相对相应参考的增量能支持该特征轴。保留所有结果与失败，单次source coarse过规则也不能自动授权小时阶段／未知区／原创机制。

substantial_context_signal只汇总History−Season、Trend−History、Size−Trend的相邻同学习器信号，避免把Season效果冒充历史／总量效应。source_choice只记录固定比较信号，不挑一个赢家冒充确认；hourly_stage_authorized/new_holdout_authorized/mechanism_breakthrough均false。是否另立小时合同取决于跨区信号及oracle局限，绝不直接套用V29小时覆盖门槛。

## 审计与后续

三份最终预审绑定5文件后才formal。独立actual auditor不导入V30producer，重构visible cohort、as-of context／feature、训练label排除、70模型/scaler/投影及全部coarse评分和133对比；首尾hash不变，重fit不是只接受storedcoeff。无需导入小时评分／路径helper，不读取raw fine。最终科学文稿批准后才新增证据清单，保留正式首终态和失败历史。

本轮只检验已有信息的可利用性。普通上下文模型有效时先留作强基线；无效亦不证明月统计无信息或必须本地小时标签。自己的风险机制、同预算强参考下明确小时效率增量及冻结后的独立地区确认仍未完成，长期目标active。

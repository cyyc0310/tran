# V17：本地滞后信息包的完整源区pilot

2026-10-25，Sol。信息设计实验，不宣称新算法。仅七源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS、2025H1同1,145个有效源地区日；没有目标/未查看区入口。V7–V16冻结文件与产物不修改。所有代码、测试、协议须先独立复核批准，之后才prepare/predict；全部预测先封存再评分。

## 四种观测包与共同可用时点

这里G是八正值燃料proxy发电能量之和，不是V10带符号BA netG。令H=Σ_f(EF_f/1000)×fuelenergy_f，EF固定[1000,400,800,0,0,0,0,400]。

| 包 | 每日原始标量 | 开放的推导信息 |
|---|---|---|
| D2 | 当日总H、G | 日加权碳锚及G |
| D8 | 八燃料日能量 | 全部fuelshares、日G、日加权碳锚 |
| T8 | UTC 00–06/06–12/12–18/18–24四块各H、G | 全部四块加权CIF、Gshares、日G和日锚 |
| G5 | 日H、四块G | 日锚、四块Gshares和日G；关键分母时序控制 |

八个标量只表示发布包维数，不表示相等Shannon信息、观测成本或现实渠道。T8可反算6小时加权CIF，所有T8直接基线均能使用，不能藏起这些强特征再把收益说成算法贡献。观测含过去碳信息，不以“没有名为CIF的字段”称为无本地碳反馈。

隔离prepare聚合器明确读取注册源2022–2025H1小时燃料值，仅将完整有效UTC日压为四个包；四包共用完全相同的缺失日mask。预测模块只读这些包、源2024允许的fine监督及旧native归档，禁止读2025 fine轨迹。聚合器确实读取评价期小时值，不能把它写成全流程没有读源评价小时标签。日包仅在day_end+7/30日之后可用，模拟滞后，不是经认证的真实发布vintage。主lag7，lag30全表仅压力诊断，不后验换lag。

## 固定前缀摘要与锚

每个origin只取已发布包，以最新可见day为末端的最近28个calendar days；缺失日不插补为真值。固定EWMA衰减14日，提供latest、原始能量EWMA后推导的全信息通道、通道weighted std、同星期EWMA（没有同星期回全部）和共同日历sin/cos。G用可逆log1p表示，fuelshares/blockratios/Gshares均完整保留；可从通道恢复原包的全部标量。当前尚未释放的包先按时间剔除，再转数值。

四包共用锚=1000×EWMA原始H/EWMA原始G，预测构造时对每个query及proper/riskfit/riskpool的同region/origin/lag逐行核对D2基准，最大差写入审计且超过1e-9即失败；不是对每日CIF比值直接取均值，也不是等权24小时CIF真值。模型以该锚为共同参考预测24小时fine残差。完全没有已发布历史时保留预测日，锚500+明确missing indicator；其他通道仅中性占位，不冒充观测。训练期和查询都使用同一规则。

## 拟合、风险学习及独立残差池

外层held源从全部global point、riskfit、riskpool fine监督剔除。held源过去已发布包作为允许本地上下文。proper为2024Jan25–Jun30；riskfit为2024Jul1–Sep30；riskpool为2024Oct1–Dec31。逐期只使用实际完整有效fine日，记录全部源region-day数和空区，不捏造缺失标签；每期至少两个实际源区，否则失败。源评价仍保留七区共同支持，不能因历史包缺失删评价日。

四个固定普通模型，每包相同预算和128个整日情景：

1. ridge_global：proper训练Ridge(alpha30)，每小时响应为fine−日锚；同样的proper-only加权StandardScaler。riskpool完整24小时残差按等地区global权重抽样。
2. ET_global：proper训练ExtraTrees(150trees,minleaf10,seed17023)，同样global riskpool。
3. ET_hetero：与ET_global共享point。仅riskfit errors拟合ExtraTrees(150,minleaf10)的24小时log(error²+25)，scale=exp(prediction/2)。riskpool按各自预测scale标准化，查询乘query scale；不把pool再次用来拟合scale。
4. ET_QRF：与ET_global共享point。RandomForest(128trees,minleaf20,maxfeatures.7,bootstrap)仅在riskfit features/errors上学习split；然后用**全部**riskpool行映射每棵树leaf，完整24小时残差向量抽样。叶内等可用地区、地区内等行，平均树权重；query叶在pool没有成员时回等地区global pool。pool label或行没有参与split、Scaler或bootstrap训练；没有bootstrap重数叠加。没有scalar-QRF/CQR保证宣称。

所有supervised拟合用source-region均衡sample weights；各Scaler仅在对应训练fold(period)拟合。ridge/ET有各自残差值，四模型riskpool地区日队列完全相同。Riskfit/Pool无交叉日期，不用pool fine标签调树或尺度。128个timestamp-seeded uniforms在所有包/lag/model中共同使用，global/conditional的indices可不同；每次抽完整误差日，保持跨小时向量，不独立抽小时。没有额外粗粒子重复叠加。

## 原控制、封存及判定

旧V14 B-native作为无新增包的强背景控制，源raw/ids逐元素复制且SHA绑定旧manifest；不更改其粒子或重选出目标赢家。五倍率均为(.75,1,1.25,1.5,2)。评分先clip raw情景到[0,1000]，其有限情景均值为point，按旧规则围绕该均值扩散并再clip。使用冻结V10的CRPS、小时覆盖、宽度、IS、上下尾、level/shape MAE，经验区间为ceil order-statistic；报告全部曲线。

总63归档：7 native+7源×2lag×4packet，各非native含四模型。代码闭包旧50+本轮4；完整输入SHA、Python可执行文件SHA、环境版本及环境摘要先冻结；生成包数据seal绑定freeze。所有预测、model/cohort/可用时点审计文件及完整网格哈希先封存，score才可读取2025 fine。计算后再次核对闭包与起始父收据。排他创建，已完成或部分输出不覆盖。

每包lag7在宏覆盖[90%,95%]且最差地区≥85%的配置中最小CRPS，若全失败按违反程度最小再CRPS，固定模型/倍率名破平局；native单独选倍率。T8源信息gate要求其通过覆盖，且相对D2、D8和native各CRPS至少下降5%、宽度不增、IS下降。另必须保留T8所选**同模型、同倍率**的G5对照及全部G5曲线，避免把分母时序信息归给H时序或模型。另单独记录H_timing_gain_detected：T8通过覆盖，且比所选同模型同倍率G5的CRPS更低、宽度不增、IS下降；该项仅描述H时序增量，不用source_information_gate替代H时间归因。source选择是开发，不是独立确认，gate不授权目标/新区或算法创新。

本轮只做exact包pilot。proper-source统一尺度量化、观测噪声、经度/太阳相位及真实发布渠道尚未完成，不能称已通过；有信号才以独立冻结新轮完成这些压力测试和公平信息预算评估。晋级前还缺少同预算coarse-CIF TSFM对照（CarbonX式6h适配，非原版hour复现），以及预训练模型截止日期和潜在2025评价污染审计；exact pilot不安装或声称完成这些对照。当前全年相位只用共同日历，不假定已解决H1训练到H2风险库的太阳相位迁移。缺少真实渠道、噪声、时间精度和相位确认时，只能报告源信息设计信号。

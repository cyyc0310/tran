# V42：扩展供体的固定权重强基线比较

2026-10-27，Sol，新增发电细分和实际拟合前固定。V41库存清单SHA `b796422578980f1855fa091ab11bc160890491b4ea46a9b305f61b48d1e85c9e`为父。测试新增十个已查看开发供体整包信息价值，不预告算法创新。原七outer、42个2025月初query/128MG粒子及35评分月保持，PACE/LGEE与真正未查看域关闭。

## 源月表和共同队列

旧七源只来自`results/heldout-2026h1-data/monthly.csv`；SWPP/AVA/BANC/JEA/SC/NWMT只来自`results/external-v4-data/monthly.csv`；SCL/WACM/SEC/SCEG只来自`results/external-v5-data/monthly.csv`。固定17名单与文件归属，不从其他区域补供体，不读取新增供体2025细分。

当前监督月2022-01至2024-06，期末和release均≤2025-01-01。先以白名单/整月/唯一键/发布时间检查metadata，再按原V30规则取POS>1e−12且有三条在该月起点已结束、已发布的过去coarse上下文。缺三条或当前公开POS近零按固定原因跳过，不补月；只转换选中上下文的总量和核/水/风/光份额，当前G5亦只用这些零因子份额及POS补集。

将V30 build_cases的固定源白名单参数化为旧7或固定17，复用相同helper/特征顺序及时间规则，不修改旧代码/SOURCES。原七源180键及其cohort features、日期必须重现。每outer的旧六源键、y4与原SizeCap-Ridge raw X前31列、日期应精确一致；若向量化舍入不同，特征固定atol1e−12/rtol0并明确记录，键/日期/标签仍exact，不按结果放宽。

只有排除outer并限定所需训练键之后，才读取四个POS细份额响应。对当前月另读同键总量和coverage作为源响应/口径诊断，不进入Size特征。coverage须在[0,1]、当前总量有限正；不因coverage低而新增过滤，避免改变旧180队列。记录现有质量；后续质量干预须单列。本轮不声称这些代理是完整官方月报的实测排放。

## 库存软特征和口径诊断

用V41固定17域OP表，每源月m取m.year−2且release≤m。记录源POS MW、四容量份额、过去最近POS MW/C_POS压力，以及current POS MW*y4与C_f=0/G_f>0矛盾。小分母、大于1的名义率不裁切、不填epsilon。矛盾只表明率模型无定义/范围不匹配，本轮六头均不做G_f/C_f或masked zero support，所以不据此删标签、源或容量，也不宣称零库存是物理缺席。V41合同允许POS库存为0；本轮SizeCap等头另有独立有效性门：任何所需源年C_POS非有限正则整次V42失败、保留attempt，不能删供体/月或加epsilon，也不悄悄对B0作同样过滤。若要B0单独继续，需另锁不同cohort设计。

## 六个固定学习头

仅扩展Size-Ridge/ET、SizeCap-Ridge/ET、DirectEF-Ridge/ET，与V34相同定义。Size31为B0；SizeCap36追加四容量比例和lag0 POS容量压力；DirectEF用同36输入。查询直接使用已封V34 Size/SizeCap/DirectEF原raw特征及MG，不重新估计粗组成。

每outer训练排除其自身全部fine标签，实际合格源各分同等总权重，再源内等分。总权重质量固定旧N=154，FPL outer156：w_i=N_old/(R*n_region)，R为该outer实际有共同队列的源数，不强造16个有标签源。不把样本增加引起SUM正则变弱当数据覆盖收益。

Ridge参数alpha=1、fit_intercept=True，权重均值/总体标准差scaler，scale≤1e−12置1，与旧规则相同。ET保持n_estimators=128、min_samples_leaf=2、max_features=1.0（float，使用全部特征）、random_state=17023、n_jobs=1，raw特征不缩放。ET叶样本数变化仍属整包数据干预，不声称纯N因果效应。

比例头训练y4，预测后按V30原Euclidean simplex一次投影，再乘POS因子[1000,400,800,400]；DirectEF训练(y4·EF)/1000，预测clip[.4,1]后乘1000。记录raw/projection/scaler/训练键、权重与Ridge参数或ET feature importance。七outer×六头=42个新fit，不训练其他架构、不调参。

## 原强参考和评分

每query保存Legacy六臂（V34原粒子literal copy）与Expanded六臂；同粒子统一计算mean/median。原1680条V40 case参考numeric exact复制，新增12臂×两摘要×35月=840，合计2520 case、504 region、72 macro、504去单源敏感性、42 support。Legacy mean若与旧存point有细微舍入差，应记录，不能冒称原point逐bit复制。模型名用Legacy-/Expanded-避免与旧参考碰撞。

全部七源预测、训练/口径诊断和参数先seal，再读取原35个2025评分标签。重点逐头Legacy/Expanded、B0/B1分开，EF/CIF都报，Capacity-OP锚及V36门控保留原强参考；不能只报告扩展后的最佳头或相反误差抵消的CIF。均值与中位数、逐源/去单源全保留，数据增益不等于方法原创或冻结资格。

## 执行和验证

固定五个新代码/测试/合同、两份独立预审、三月表/库存/原训练与query/参考及代码helper输入hash后prepare。exclusive目录`results/donor-budget-v42`，started先于新fine响应读取；失败保留、不重跑失败attempt。预测目录七NPZ+七JSON；cohort metadata.json不含fine标签。完整seal校验grid、shape、hash、MG乘法在score标签前完成。

独立临时公式复核队列/原180、源标签与权重质量、42个新fit、投影及全部粒子/2520评分，预锁atol1e−9/rtol0（原特征回归1e−12），原1680参考numeric exact；ET独立同配置重fit属于验证不另运行正式stage。源scope矛盾只诊断，不能变成过滤或隐藏另一个预算。不得另造大型审计框架、改原封存文件或读新增供体2025fine/原小时/保留区域。长期目标active。

# V184准备：区分更长历史输入与更多粗监督

2026-10-29，Sol，待实现设计，不是新结果或独立确认。V181主评分已查看，完整审计仍在进行。精确独立配对对V180对角CRPS仅约0.92%改善，未过覆盖/效率gate，不直接发展复杂copula。V183已隔离导出661个粗历史月且327个原2024 profile精确复现，先检验该新增信息是否真正有价值。

## 固定研究问题

当前V178月度分布森林用原细世界名单训练，2024早期又因没有更早前缀而缺少合法月度训练上下文。增加粗监督可以不增加小时训练样本，但所有基线必须获得相同信息。更多数据不是原创，先测价值再设计风险机制。

仅使用原29源、六折、2024 asof_30d、相同241个test月。不打开2025或未查看地区。本轮先做粗组成分布，完全不读取源小时数值或评分小时CIF。原小时gate不变，粗指标通过不能晋级新地区。

三种预算：

1. short：原MonthlyInformation，训练名单精确取V178该fold原全月细世界profile ids（只是名单，不读g），base13/history36仍由原函数构造。
2. long_fixed：追加V183历史，训练标签profile ids与short逐个相同；只让输入前缀更长，隔离历史特征的作用。
3. long_all：同一长前缀，同时允许proper源2022–2024全部有合法输入的月标签。2024 Jan–Mar可因更早前缀获得合法上下文，且增加2022/2023监督。不可训练cal/test地区的全局模型。

三者都保留当前共享2024源开发合同，不称整条小时框架已前向时间验证。若另加2022/2023训练、2024评价，必须独立固定为粗层时间外推检查，不能混入本轮三预算。

## 数据容器与隔离

新ExtendedMonthlyInformation只读原margin/index及`monthly_history.csv`。原936个profile id及M/shares保持bit-exact，历史记录在index尾部追加，不能重排原id；groups/local按日期而不是id选最新合法输入。合成发布时间必须逐行等于月末+30d。2024 QA/质量JSON/原始CSV/小时generation不得进入该容器。

模型的query特征始终从release≤origin的prefix重建；源proper当前月量只作为监督。query当前/未来月量扰动不能影响任何预测，做12个首尾test query检查。预测构造完成前不读取query当前组成评分。数据文件只包含原29源历史，不假定中国或未查看地区有相同覆盖。

## 同预算强对照

每种预算九个分布方法，另有一个公共点持平参考，共28个粗方法：预计241×28=6,748粗评分、53,984燃料评分。实际分母在runner实现时须核对，不静默丢月份。

- persistence_residual：点为最近合法S；proper月度持平误差ΔS作为联合残差分布。
- seasonal_residual：若合法prefix含上一年同月，取最近该月组成，否则持平；proper同规则回测误差构造联合残差law。
- trend_residual：最近最多12个合法月，至少4个月才用带截距的Ridge(alpha=12)线性月份趋势外推八类组成，否则持平；投影simplex。仅拟合本地合法粗历史，允许query已公开月量，不用query小时标签。
- pooled_linear_residual：base13标准化、proper地区平衡权重的多输出Ridge(alpha=1)拟合ΔS。proper五地区折OOF残差加到query点上，再simplex投影；每个OOF fold独立拟合特征标准化，不能使用被留出地区标签。元训练样本权重归一到训练样本总数。
- gaussian_increment：base13、32树/leaf4/seed20261028的标量ΔEF点森林；proper五地区OOF RMS给σ≥1，沿用原past-single-EF点fallback。对本预算加性ΔS粗支持，pH乘exp(−.5*((v@EF−Ahat)/σ)^2)，不能仍对旧乘性支持算Gaussian。这里是普通标量不确定性强基线。
- forest_base_scalar8 / forest_history_scalar8 / forest_base_delta8 / forest_history_delta8：与V178相同32树、leaf4、max_features=1、seed、地区平衡经验叶概率；响应proper标准化、std下限1e−3，Δshare/ΔEF按既定64eps规则消去数学零噪声。每法支持是query过去S＋本预算源ΔS的simplex投影，概率=.95叶概率+.05相同pH，pH用原bw=.3的Hellinger核及源地区等权/月等权。不扫描树容量、先验强度或RFF。

点＋残差法的残差均来自本预算proper标签名单；persistence/seasonal/trend回测只用各源当时已发布prefix，无全局源模型参数。所有残差经验概率按proper地区平衡。五OOF地区折固定hash/region排序并记录，避免标签或表顺序变化改变划分。普通趋势、残差/QRF、Gaussian与长历史均非原创。

## 必须检查的负控制

在相同原训练profile ids及query月份下，short与long_fixed的base13、源ΔS和query最新S应完全一致。因此base响应森林、pooled_linear及base Gaussian的训练/预测应复现；history36与季节/趋势法允许变化。若这些base控制变化，先查id、数值单位、角色或额外信息泄漏，不能当作历史收益。

公共点持平参考应复现V178 coarse_persistence的241月评分。原Gaussian/全月Gaussian、V179加性对照的粗指标可列为短预算外部参考；长预算候选优于它们不等于优于同预算强基线。

## 评分、审计与判断

沿用完整组成energy score、同EF组投影ES、八边际CRPS、月零Brier、EF锚CRPS及原始90%区间覆盖/宽度。月份等权再地区等权，all29及nonconstant24都报；不要把粗指标解释成小时CRPS或条件覆盖保证。报告更多粗监督/历史对同一算法的配对差异，以及长预算各算法之间差异。所有bootstrap仅为反复查看的源开发描述。

保存每预算源标签/prefix/角色、标准化与模型hash、query支持/概率及残差来源；全拟合及全部粗/燃料评分和汇总重建，含独立loss式与12未来数据干预。质量构造共享时如实说明，不称真实发布证明。

若长历史及更多粗监督对强基线都无稳定粗增量，停止往小时层扩大模型，回查数据和可识别假设。若确有粗增量，先用普通传播和强对照检验小时价值；有完整小时覆盖和效率增量后才冻结新机制、研究原创性和真正未查看地区。

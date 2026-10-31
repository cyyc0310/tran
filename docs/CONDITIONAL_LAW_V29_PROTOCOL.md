# V29：固定长粗监督，比较组成条件分布与异方差普通强参考

2026-10-26，Sol。运行前合同。上一目标轮有实际progress：V28完成正式三阶段、独立294模型重fit、全路径及评分核验与科学／证据清单审定。长源粗统计明显有效，但复杂重组未超越同预算简单锚，且全部105离散配置不合格。本轮执行下一必要强控制，不能把普通概率模型称为原创机制。

## 输入、源区与时间

父证据`docs/evidence/coarse-supervision-v28-verification.json` SHA256 `18cbd15d32d76801c0467b43370916c582237b6b45014a11971f8918a32c2a51`及其完整输入／代码／产物闭包固定。沿用七个已查看源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS、2025H1开发起点，共同1145有效源日；不打开PACE/LGEE等旧目标或真正未查看地区。

用户已确认可再生与不可再生电源各类月发电量，按完整同口径月类别预算；仍不自行假定火电内部煤气油细分。查询使用原MG五组（POS＋核水风光）粒子，不输入隐藏燃料比例、小时发电或CIF、天气、年度细分。美国代理类别／总量边界与模拟发布并非中国字段或真实vintage认证。源月表虽有total_mwh等，本轮只用原五维X，不加绝对量／历史／coverage特征，完整公开月量的信息价值尚未穷尽。

仅L24粗监督窗：2022-01至2024-06，部署截止2025-01-01T00:00Z，沿V28字符串先排outer全部历史、再按期间和release合法筛选后才转数值。各teacher fold排除其全部历史后重新训练树结构、叶响应池和权重，不可用六源树再删teacher标签。查询日期不得早于固定截止，standalone也须在模型predict前拒绝；不能称2024逐日合法forecasterror。L24由V28已查看开发结果选定，不是独立预注册窗口发现。

源监督y为正POS四维比例r，X为POS/核/水/风/光组成；POS=0排除r监督并记录，重复region-month拒绝，不补月。proper小时q、bank月响应与采样池沿V28固定2024H1，不增训练fine解析。bank标签是旧eligible-support月定义，不偷偷换完整月粗标签；粗监督与bank定义差异继续披露。

## 森林权重和三种新普通分布

沿V28 ExtraTreesRegressor(n_estimators=128,min_samples_leaf=2,max_features=1.0,random_state=17023,n_jobs=1)，不调新网格。每次fit独立重算`a_i=N/(R*n_region)`，每地区总质量相同、区内月等质量、权重均值1。每outer一个六源final与六个五源teacher，共七森林；七outer共49新森林。所有训练X/y/keys/期间/release/weight/leaf与排除信息留档。

对于任一拟合队列，树t的query叶L_t(x)与训练叶匹配：

`w_i(x)=mean_t[a_i 1{L_t(X_i)=L_t(x)} / sum_k a_k 1{L_t(X_k)=L_t(x)}]`。

所有权重非负、和为1，μ(x)=Σw_i r_i须复现该ET条件均值至1e−12；final μ须同V28 L24 ET raw query均值至1e−12。树结构完全相同但叶经验分布不是原始经典RF设置或MMD分裂DRF的严格复现。

1. **Leaf**：用独立固定uniform按w抽整行四维r，天然simplex，保留燃料联合关系，不独立抽边际分位数。
2. **Gaussian**：从同一跨树混合后的总w计算完整四维组成协方差。固定4×3 Helmert H满足HᵀH=I、Hᵀ1=0，Σ3=HᵀCov_w(r)H；取唯一对称PSD根V diag(sqrt(λ)) Vᵀ乘独立normal3，再映回r4，最后一次simplex投影。允许退化，仅守卫数值小负特征值，不缩减协方差、不按评分加膨胀。必须包含树均值间散布，不能仅平均树内叶协方差。投影会改变实现均值／协方差，与Leaf之差不是纯高阶矩因果消融。
3. **Scale**：同w定义组成条件散布`s²(x)=Σw_i||r_i−μ(x)||²+1e−12`。这里是**加性数值regularizer**、不是max硬截断，更不是折外误差协方差。每bank月j用严格teacher五源的μ5(X_bank,j)、s5(X_bank,j)，响应r_bank沿旧银行。查询有效供体j给

   `r_pre=μ6(MGdraw)+[s6(MGdraw)/s5(X_bank,j)]·[r_bank,j−μ5(X_bank,j)]`，再投影。

   scale与残差、q/support同有效j联动。方差regularizer固定，不看2025结果更换。记录query/source s、尺度比、regularizer占比1e−12/s²、近floor标记s²≤2e−12、投影前后偏移。近零源散布可能放大残差，按结果报告，不裁尺度比来补覆盖。此法是普通位置尺度迁移，不声称其抽样精确等于叶经验CDF。

Leaf/Gaussian采用日期稳定seed17023与固定salt的独立新随机流，保存uniform/normal3，不改变旧MG/q/IDs采样，不复用相同随机比特伪装独立。均值、cov、scale按query粒子计算；没有2025目标条件参数拟合或小时端校准。

## 十臂、配对及缺源燃料

固定顺序：

- 背景：`L24-ET-plugin_recompose`、`L24-ET-residual_anchor`、`L24-ET-residual_recompose`。八字段逐数组exact引用V28归档，不只比较均值或分数；旧模型只继承父审计，不冒称再次重fit。
- `Leaf-anchor`、`Leaf-recompose`。
- `Gaussian-anchor`、`Gaussian-recompose`。
- `Scale-anchor`、`Scale-recompose`。
- `Scale-shuffle_recompose`。

同family锚／重组共用fine_draw和有效q/support；fine_POS=MG_POS·r，非POS原样。锚沿旧公式`raw_CIF_bank[i]+[(fine_draw−mix_bank[i])·EF]`，原i不换成fallback j。重组按fine_draw与q逐小时能量比值，不擅自新增未来实际总量约束。

POS缺源时沿原fallback：POS残差所属bank月、q与support用有效j，非POS q/support保留i；正POS无合法有效供体则拒绝。query POS=0不得误fail。新独立r可给未支持fuel质量，q=1仍是显式平坦先验，逐粒子保存unsupported POS能量比例；这不代表该fuel小时风险已有监督。

shuffle在**同query的fallback完成后**，对整24×8有效q及support一同做固定置换，Scale fine_draw保持逐粒子不变；保存置换及置换后原i/有效j provenance。fine与q各自输入多重集保持，输出CIF边际并不必相同，且破坏的是完整fine−q配对，不仅隐藏比例copula。zero_total_mask及其创建率单列，不使用shuffle通过率宣称纯依赖因果效应。

独立Leaf/Gaussian组成取消原q残差关联。因此各family对比同时涉及条件分布／中心／投影／依赖，不将其简单称“只改变方差”或新风险机制。组内EF越源标量范围不等于四维凸包或完整小时路径支持。

## 输出、评分与固定比较

181天×128粒子×24小时、10臂×五倍率0.75/1/1.25/1.5/2=50曲线。所有七预测封存且验证完整库存后，才读取已查看七源2025H1小时fine评分；没有新的小时训练解析。沿用初clip→point→spread→clip、CRPS、90%经验区间及所有旧指标，按区日均再七区均；共同1145日产生57,250日／350地区／50宏行。

固定16对，每对7区×5factor=35行，共560匹配对比：

- 背景残差重组分别减背景残差锚、背景插件重组，共2。
- 每个Leaf/Gaussian/Scale：新重组减同family锚、新重组减背景残差重组、新锚减背景残差锚，共9。
- Leaf重组减Gaussian重组、Leaf锚减Gaussian锚，共2。
- Scale重组减Leaf重组、Scale锚减Leaf锚，共2。
- Scale shuffle重组减原Scale重组，共1。

归档八通用字段raw/fine_draws/ratios/unprojected_ratios/projection_l1/unsupported_positive_energy_fraction/source_EF_outside/zero_total_mask，原MG/IDs、query与训练叶ID、查询行权重、抽样r索引/随机量、μ/Σ3/s、bank折外μ/s/残差与fallback映射。无隐藏query粗响应进入模型。

宏覆盖90–95%、worst≥85%仍原门槛；未合格fallback保持false，不后验新增1.2或放宽95上限。报告固定倍率完整结果与每臂旧源规则选择，不能只展示有利地区。source coarse拟合／分布散布不带来小时覆盖保证；本轮没有独立小时校准集或新条件覆盖理论。全部2025响应仍是反复查看的开发数据，不能改名独立确认。

## 执行、审计和路线判断

三预审通过且绑定最终五文件后才prepare；prepare/predict/score独占写，失败尝试不覆盖。冻结5code与完整父闭包／3新预审输入，七预测全seal后才score。独立实际审计不得导入V29 producer，重建源队列、全部49森林、全局叶权重／矩、teacherμ/s、随机流、simplex/有效q/置换及全新路径；旧三个背景八字段与父exact比较。完整50曲线／560对比／选择与预定Jan/Apr首valid日显式pair CRPS核对，首尾hash不变。科学最终解释批准后才封证据清单。

本轮是普通联合条件分布与异方差强参考，不宣完整DRF、全部条件分位数路线或创新成立。若普通参考已解释收益，保留其为强基线；若只靠增宽、简单锚仍优或最差区失败，不能称原创新门槛达到，也不能推出月信息无用或必须新增本地小时标签。自己的机制须在合格覆盖下相对同预算强参考CRPS至少下降5%、宽不增、IS下降并有独立确认；普通参考合格本身不自动打开未知区。真正机制与未查看地区确认尚未完成，长期目标保持active。

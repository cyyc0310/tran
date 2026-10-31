# V49：严格排除两地区的源小时风险数据

2026-10-27，Sol。父V48清单2790acfbf96e24b37d1baa2f45a6431ccf247a8e8f71c482d0eab114cea59b64。V48已定位日均与形状风险的作用，但固定曲线事后取点不是风险校准。本轮先建立可用于后续global/component/异方差/条件分位数比较的源风险数据，不训练新风险方法、不读2025小时评分、不打开PACE/LGEE或未知地区。

## 数据与排除合同

只用原七个已开发地区。风险日期固定2024-01-25至2025-01-01前的342个UTC日（H1 158日、H2 184日分别报告）；每个outer评估地区o和另一held地区j构成有向任务(o,j)，共42。生成所有42×342=14364日预测，即使某日无合格标签也保存预测。标签支持在准备阶段只由hourly CSV的time/region/valid元数据固定：唯一、UTC整点、24小时齐全且valid全True；不先转换generation。实际source_days有效集合若不符则整次失败，不按成绩删日。

21个无序pair的bank直接取旧pair-excluded缓存。要求excluded精确等于{o,j}、source_regions及实际region集合精确为其余五区，row_ids、日期、cutoff等符合原H1 bank：2024-01-25至2024-07-01前。hourly q/support/rawCIF/donor_mix均来自这五区。不能拿原六区bank只改响应mask，也不把有向两任务当两个独立bank重新拟合。每pair共享模型，分别预测o和j；该pair的两个地区细标签均不参与任何训练。

Scale比例模型沿V29固定算法：完整月度fine监督2022-01至2024-06、Jan2025快照，先从rows排pair再转换响应；conditional ET128/leaf2/max_features1.0/seed17023/n_jobs1。final拟合其余五区，每个bank donor地区k的残差由再排k的四区模型生成，共6个ET/pair、126个。bank月比例是proper bank observed-support定义，非完整月细比例，差异保留；条件s²=traceCov+1e−12、scale s=sqrt(s²)，倍率为s_query/s_OOF。不得用bank donor自己的fine训练其OOF条件均值。旧strict六区入口不改，新入口显式支持pair排除。

另每pair重拟合两个固定月度中心：B0 Legacy-Size-ET与B1 Legacy-SizeCap-Ridge。训练记录取旧V30/V34，先按key/region/日期排pair，才转换X/y，label仅2022-01至2024-06且release<=Jan2025。原31/36列及超参数不变，五源等质量、总权重为剩余N，Ridge train-only weighted scaler；21 ET+21 Ridge。合计168 sklearn fits（147ET+21Ridge），两个方向共享。源五区到实际六区预测存在算子规模差异，不隐瞒。

## 合法输入与预测定义

每个源j的342日MG只依赖j截至每日零点已发布的粗月表，四零EF份额与POS补集；不读取j历史内部coal/gas/oil/other。原V25 own-past harmonic/prequential law、128draw/seed17023，七源各生成一次，跨outer共享。按完整发布时间<=query，不floor release。每日Size特征同V31公式，month phase/月龄保持整数月，release_age按真实秒差；B1额外用2022 OP库存（query.year−2），不是2023或真实出力硬界。

五源bank按原MG conditional donor_weights和同日原uniform RNG抽donor/fallback/q；用final条件均值与scale、严格四区OOF residual生成Scale ratio并投影一次，构造fine。原anchor A及燃料重组R按V29公式，U=fine·EF，P=MG_POS*(query_mean4·EF_POS)。不能为迁就旧2025入口monkeypatch日期常量或伪造background数据。

每个任务固定8个base：Scale-anchor、Scale-recompose；B0 SizeET与B1 SizeCapRidge各3模式：Keep-A=A−P+S、Replace-R=R−U+S、AlignedKeep=R−mean_hR+mean_hA−P+S。新增中心S=同日MG_POS乘源五区月度头EF，全部同日同粒子。8base预算前5为B0（原2+B0三），后3为B1。原已封2025查询预测不改，本轮不把新五区teacher当成原六区预测literal重放。

保存每源MG/合法context与features31/36，各任务raw8×342×128×24、level2×342×128、U/P、donor/fallback、fine及日期/键；保存21pair模型审计与训练key/排除证据。新Scale适配器先做六源synthetic同原V29算法parity，不用新真实标签做调试选参。旧producer代码不改。

## 标签、误差与后续使用边界

prepare→predict→label各一次，exclusive results/source-risk-v49。所有21pair拟合及42完整预测先seal，label_started后才按原七源2024-01-25至2025-01-01窗口转换合格generation并算固定EF代理truth；每源truth只保存一次。每任务只保留预锁支持，逐base先clip raw到[0,1000]求point。保存signed level error=mean_h(truth−point)，shape error=center_h(truth)−center_h(point)，全小时error须等于level+shape且shape均值0（tol1e−9）。不把日算术均值当月能量加权CIF，不在本轮选scale或风险模型。

元数据盘点已固定有效日BPAT180/CISO181/ERCO69/FPL180/ISNE26/MISO182/NYIS182，H2共1000源日。另H1有效日BPAT157/CISO158/ERCO158/FPL153/ISNE158/MISO158/NYIS158，共1100；总2100源日、12600有向任务有效日、100800个base日误差。ERCO和ISNE在Q3均无标签，前者只有Oct8/Nov30/Dec31，后者Nov2/Dec24。不能默认Q3/Q4拆分后每源均有资料；不填补或删除这些源。后续风险拟合/校准划分另预锁，本轮只保存日期，不假定iid/跨区conformal保证。

各outer的风险训练任务集合必须恰好其余六个held；所有teacher/head/bank均排outer，不通过别的teacher回流outer fine。若额外做meta源外验证，被验证源必须从其它teacher中再排除，不能直接拿此双排除表声称严格第三源验证。保存严格按outer选择六个有向任务的metadata索引，禁止把反向任务含outer的标签并入。

H1是任务级地理交叉拟合：某源的H1样本可以作为其他合法任务的训练bank数据，不能声明这些数值在全局预测前从未出现；但该源被预测的任务必须从bank/比例头/中心头中全部排除它。全预测seal保护的是风险响应接口与调参顺序，不是全局H1数值新颖性。H2标签不在H1 bank中。

此数据为Jan2025点模型训练快照的源外回溯预测误差。H1仅作地理排除和同季节诊断，其他五源的bank与fine头可能包含晚于H1 query的材料；H2仅保证query日历晚于H1 bank，fine头仍可能用早期H2 origin尚未发布、但按旧监督元数据在Jan25快照可用的监督。两个panel都不伪装2024真实在线或完整as-of时间留出。own-MG与context仍严格按每个origin的合法粗前缀，不能因回溯而放开。未增加目标本地信息，不代表中国完整B1可得。重复源日期跨outer出现不等于独立样本。

风险hourly标签的实际首发/修订时点未认证，观察日期本身不证明该标签在Jan1已发布；特别不能自动将Dec31样本当作Jan1已知校准响应。本轮只生成回顾数据，不拟合风险方法。索引明确hourly_label_release_policy=null、actual_first_release_verified=false、requires_release_policy_before_risk_fit=true；后续必须另锁标签发布预算或延迟敏感性规则，并按每个部署query日期筛选，才能训练和比较风险方法。不能将旧点模型fine监督的available_at规则直接套给这些hourly风险标签。

## 验证与输出

7个新producer/tests/protocol（pair-scale、centers、runner、三tests、本合同）及两实际落盘预审、父manifest/直接依赖预锁；只解析父1134 metadata并核本轮直接文件，不重遍历旧数据。数值仅freeze/prediction_started后读取，标签仅完整prediction_seal/label_started后。失败保留，不因观察超时重启。准备阶段支持数量由元数据落盘，不写猜测计数。

保存100800个base日评分及按panel分开的672个outer/held/base源均值、112个outer/base等held权重汇总作诊断（先每held按各自有效日平均、再六held等权，不把H1/H2混池掩盖shift），不用于选择方法或尺度。独立临时公式重建全部21pair/42任务MG/point heads/conditional forest/OOF残差/抽样与raw8、合法历史context、2100个valid源日标签与误差分解；168验证fit及官方首次168实际fit分别记录。核所有mask/key/源计数、相同pair双向复用，raw/数值tol1e−9rtol0。不得只核少量日期便说全量通过。若源风险关系或支持不足，这也是下一轮方法决策所需的结果；仍不冻结或开未知地区，长期目标active。

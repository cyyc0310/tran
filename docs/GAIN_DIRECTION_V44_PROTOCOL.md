# V44：强端点源外增益方向诊断

2026-10-27，Sol，新增source MG与拟合前固定。父`docs/evidence/ordinary-controls-v43-verification.json` SHA256 `a98874cbccf7e1f9d916de4bce15bc27a1abe79d4bdcb86f505c7b2369644ab6`。七个2025评估地区已反复开发，42查询/原35评分和128原MG保持；所有结果仍属开发性实验。PACE/LGEE、真正未查看域、原小时及新增供体2025fine关闭。

## 问题与端点

检验预测时合法输入能否预示两个强端点之间的增益。Base为原CoalOilUpper，候选分别为V42 Expanded-SizeCap-ET、Expanded-DirectEF-ET；是整包预测增量，不声称纯容量因果效应。保留V43全部4970参考，包括强B0 Size-Ridge、两候选和固定普通混合14.7219/29.7641参照，不对弱匹配ET路由继续调参。

执行前只读旧case的事后潜力检查显示：Expanded-Size-ET/SizeCap-ET逐月硬路由的不可用CIF oracle下界16.2052，高于普通混合14.7219，故不投入该对。强Base/Direct候选的单独CIF/EF oracle分别11.2920/8.3514，但按EF选同一个action得到CIF16.6547，不能把两个单独oracle合称共同可达收益。这些缓存标签仅用于开发性问题选择，不是独立确认。

## 412源月、粗信息与合法teacher

fine/head池原412去重源月、16合格地区保持；每outer先完全排除其fine，剩15源386条，FPL388条。训练截止2024-06，所有标签2025-01-01前可得；不改质量或删月。复用已封V42各outer训练数组，metadata排除优先于数值消费。

source MG按每一个412源月起点，从该地区期末和release均不晚于起点的own粗历史生成。同样是V25五维harmonic/prequential残差law：128粒子、seed17023，日期seed键低/高32位、年/月/日；最近60月point、24残差、固定原罚项/权重。用薄generic入口仅扩固定17名单，复用封存V25 _point/_ordinal/_simplex_rows等数学helper；不monkeypatch名单，不call会转换八维fine的group_monthly/monthly_prefix。先筛合法prefix，再只取核/水/风/光并以补集形成POS，不转换当前或未来粗值，也不读内部细分。每个残差的history同样必须先于其pseudo起点且已发布。每source-month只生成一次，同draws跨outer/inner/candidate共享。

预检合法prefix3至28条；原law在少于12条时用最近同月/最近月，空历史固定prior，无需删412。记录prefix/residual数、零历史/不足12条残差及实际harmonic solve/fallback/prior次数；冷启动不认证校准。粗信息生成确有自身估计工作，单列，不称全阶段无coarse估计。原2025 query MG仍逐值复制，不重新生成。

## 双排除细头与增益标签

7 outer各对15 inner source留一，共105折。每折base Ridge只用旧七源中排outer和inner后的5或6源；两个candidate ET用全部合格地区排outer/inner后的14源。所有source-month为同原412池的相应子集，不把已看inner细标签的V42 head直接拿来当teacher。每折3fit，共105Ridge+210ET=315新细头拟合。原alpha1、ET128/leaf2/max_features float1.0/seed17023/n_jobs1，权重各源等质量、源内等分，总质量154（outer FPL为156），scaler和投影沿旧定义。

source query把Size31前5个真实当前G5替换为上述128MG，其他26个合法历史/季节特征保留。Base Ridge先预测ratio并投影，再按该折旧源q的CoalOilUpper选择容量锚；candidate分别用SizeCap36比例ET、同36 DirectEF ET。q和C_POS仍按source origin.year−2登记库存，旧q=0不作物理缺席保证，均不做G/C_f或填epsilon。source query的fine响应只用于其预测完成后的损失；不得进产生该预测的拟合。

用同128粒子的mean EF和mean CIF对source真值计算共同端点的三个增益：D_EF=(|eL−e*|−|eR−e*|)/600，D_CIF=(|IL−I*|−|IR−I*|)/1000，D_Joint=.5*(D_EF+D_CIF)。true EF由source y4和固定因子得到，true CIF由source当前真实POS份额（非MW）乘true EF，仅作标签。joint是预锁标准化效用折中，不保证两项同时改善；teacher都基于mean，不混用median。

这是2025-01部署快照下的回溯跨源teacher：MG输入在各pseudo起点合法，fine头则用Jan25快照下其他源至Jun24的监督，不冒称2022–24当时真实历史版本前向回测。14→15源candidate和5/6→6源base的operator变化仍是迁移误差来源。

## 固定meta模型与30路由

两个输入view：HistoryForecast28=Size31去当前G5前5的26维，加同source MG_POS均值和总体std(ddof=0)；HistoryForecastCap33再加q4和lag0 POS MW/C_POS压力。不用当前真G5、fine、当前total/coverage或地区ID作meta输入。outer query同样从原历史后缀、原MG和原q形成，128粒子内恒定须核。两个view整个预测都用B1端点，HistoryForecast只是gate输入消融，绝不是中国B0方法。中国目前只确认粗月报B0；库存B1可得性未认证。

每candidate、每D各5估计器：源均衡Mean常数；两个view各Ridge/ET。meta总权重6，15源均分再源内等分；Ridge alpha1、weighted scaler，ET用原128/leaf2/float1.0/seed17023/n_jobs1。每outer共24个sklearn meta拟合+6个Mean估计，总168meta fit+42Mean；加细头315共483sklearn fits，粗law估计次数另计。

固定预测D>0选candidate，否则含tie选Base，整query一个action复制该端点128粒子，不再混权重、调阈值或clip。顺序candidate SizeCapET/DirectEFET，目标EF/CIF/Joint，模型Mean/HistoryForecast-Ridge/HistoryForecast-ET/HistoryForecastCap-Ridge/HistoryForecastCap-ET，共30臂。meta预测值不称概率、证书或校准置信度。不做source内meta CV；若以后做须重建相应排除的teacher，不能用共享标签伪CV。

## 执行与验证

固定五个新producer/测试/合同、两份落盘预审、父链所需别名/helper哈希。只解析父manifest元数据并核本轮依赖，不重遍历旧941数据。exclusive目录`results/gain-direction-v44`，四stage prepare/teachers/predict/score一次，exclusive日志/started；失败保留，不因观察超时重启。两预审实际存在并核SHA后才启动。

teachers先保存412个sourceMG及诊断、每outer按训练键顺序的3端点EF/CIF粒子、三个gain标签和两个meta view、105折参数/训练索引/权重/scaler/系数或importance；全部先teacher_seal。predict再拟合meta并复制outer原端点，30×6×128，各query route/预测gain写盘后全部prediction_seal；score_started先于读取35outer评分标签，已有started拒绝重入。

原4970参考numeric exact，加30×2×35=2100共7070case，1414region、202macro、1414去地区及42support；另mean口径增益诊断1050case/210region/30macro。令a=1(predD>0)，真实D使用同臂对应EF/CIF/Joint效用；regret=max(D,0)−a*D，非劣action为(a=1且D≥0)或(a=0且D≤0)，D=0两action均非劣且单列tie，不事后加epsilon。报告predD−D及abs/sq，按地区先平均，预测RMSE地区开方后再macro，保留路由率。不把月当独立域，不报虚假的source训练内泛化分数。mean/median预测全保留，regret只针对预定mean效用。

独立临时公式验证source coarse law与prefix、315细头和168meta refit、42Mean、所有teacher/route/query粒子及全部评分，旧4970exact。atol1e−9/rtol0，键/源排除/布尔action与literal端点exact；原helper数值等价不放宽门。先核合成保护测试和代码/Python预审，不新增大型审计框架。

## 停止与创新边界

Mean是必要无条件参照；若合法view不能降低相应选择regret并优于Mean，且端到端仍弱于Coil/普通混合，应停止当前收益gate假设，不继续调权重或换名字。若有稳定增量，先完整报告逐地区/去地区、信息view消融和teacher shift，再设计专门机制与风险区间；普通crossfit/gate/regression本身不构成原创。无小时/区间覆盖或宽度保证，不改变冻结/未知地区资格，长期目标active。

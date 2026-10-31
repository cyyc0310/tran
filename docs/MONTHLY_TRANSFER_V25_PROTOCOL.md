# V25：同月度预算下的组内燃料与小时形状迁移

2026-10-26，Sol，运行前合同。用户仅有公开月／年度燃料统计；本轮只验证月度代理预算，不宣称年度接口、中国数据验证或原创机制。V24显示字段合并先损失粗碳水平识别，本轮检验普通源先验能恢复多少预测能力。

## 信息、队列与时间

七个已查看源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS，外层逐区排除其fine训练数据。proper为2024-01-25至06-30完整源日；从R1封存的两个双排除bank按region/day取并集，恢复其余六源。重复行的q/rawCIF/donor_mix/monthmean/support及月份必须exact相同，按旧query归档六源provenance排序，逐行一次，不增加样本数。训练和预测不新解析fine文件；评分仅在七区完整预测封存后读取已查看七源2025H1响应，预计共同1145源区日。没有新目标或未知地区。

唯一粗输入为原source monthly.csv修订EIA代理，available_at仍为模拟月末+30日。先按字符串时点筛掉晚于2025-06-30的period_end/available_at，再解析历史数值；每origin及每历史pseudo-origin再次筛当时前缀。只接受整月、单一冻结vintage。不得使用2025当前月真实组成、日统计、小时G、天气、telemetry或小时CIF上下文。

两轨：M8为八燃料份额；MG先将coal/gas/oil/other合并为POS，再加nuclear/hydro/wind/solar共五维，删除八维隐藏列。MG的谐波拟合、支持、预序残差与核只使用五维，禁止先预测八项再合并。固定谐波ridge、最多60历史月、24预序误差、128联合粗情景，种子17023。冷启动prior为均匀八项的共同粗化，即MG=(.5,.125,.125,.125,.125)，而非均匀五项。常规支持限制只描述既有预测器假设，不是本地物理不存在证明。

## 九个固定ordinary控制

M8和MG各有anchor_conditional、recompose_global、recompose_conditional、conditional_shuffled；MG另有conditional_plugin_split，共九臂。全部为已有类方法的透明控制，不预称自创。

global供体先验为等地区质量乘共同calendar核（带宽.5）；conditional再乘可见空间Hellinger核（带宽.3）。query origin独立确定旧RNG，128×8 uniforms列0抽供体；不能按评分改带宽或增删臂。

M8直接使用八维粗draw。MG按所抽供体的**地区月**正EF组比例r拆POS draw，其余四组直接恢复；r与同供体q联合。q来自原地区月observed-support归一化轨迹，不是日归一化，不能声称任意抽样的未来月总量严格守恒。

所抽供体i若POS月份额为0、且query有正POS情景，则在同weights限制到POS支持行后重新归一化，用uniform列2选j；r和POS组整段q共同借j，其余四组q沿i。原i的POS月量为0，因此替换该组q不改变其原rawCIF。记录i/j，不能只借r留下不支持组flat q。全bank没有POS支持且需要正POS时明确失败；query所有POS draw为0时允许零质量，不误判失败。该回退是源先验，不能当作当地硬支持。

anchor_conditional与conditional重组共享IDs、有效q、隐藏比例和fine_draw，前者为`rawCIF_i + EF·(fine_draw−donor_mix_i)`整日平移，后者为`R(fine_draw,q)`。这项匹配隔离比值算子，重组不独享隐藏燃料信息。

shuffled固定fine_draw及有效q各自完整128项多重集，仅按独立origin-seed排列q配对；记录排列。经非线性R后输出CIF的均值/宽度/边际可变，不能称输出copula-only或纯尾部消融。M8里粗draw与供体流本来独立，shuffle是有限粒子配对敏感性的负对照；MG对照也不构成跨预算因果差分。

plugin比例为同conditional weights在有POS支持行上的比例均值，不是平均mix再取比例。q和粗draw沿joint控制，移除比例随机性同时改变比例–q关联及潜在中心，不将其差归于纯方差。全组无支持的零质量处理与其它MG臂相同。

## 固定验证、评分与选择

正式前合成测试必须覆盖：MG内部燃料重分配不变性（可见五维不变、proper库固定）；M8 law与旧V14谐波法则一致；pairbank重复1ULP拒绝、六源行序；as-of未来poison；零组支持／fallback共借q+r；CDF尾零；shuffle输入多重集；plugin定义。

prepare绑定R1最终PASS manifest SHA `7a39eb59d98ce3323e443c49e3ead987813be855f49d3c21865155d71beaec8f`及其闭包、输入、代码、环境和三份预审。predict先核M8 conditional供体IDs必须等于旧query native，raw最大绝对差≤1e-9（数学重现，未承诺新存储布局逐字节身份）；不满足则保留失败、禁止seal/score。七区各一NPZ包含九臂及必要coarse law、IDs、fine_draw、fallback/permutation、bank行映射；全七档完整seal后才允许fine评分。部分失败不覆盖、不自动改变容差/模型后重跑。

固定倍率(.75,1,1.25,1.5,2)，沿旧clip(raw,0,1000)→均值→同均值扩散→clip约定。CRPS、90%覆盖/宽度/区间评分、上下尾miss、日水平与中心化形状误差沿旧评分器；每区有效日等权，再七区等权，保留逐区最差覆盖。九臂×五倍率45曲线，预计51525日评分行、315区行、45宏行。

每臂只由七源评分选倍率：宏覆盖90–95%且worst≥85%才合格，合格中最低CRPS；无人合格则保留failed fallback，按覆盖违约、CRPS、factor排序。不跨预算宣布算法胜负；MG对自己的anchor/global/plugin/shuffled及同factor比较，M8作精确信息参考。相同factor比较与选型比较分开；不同配置、粒子、重复月份不增加独立样本量。

本轮是ordinary基线与信息合同验证。即使joint优于global或shuffle，仍需完整同预算直接回归、异方差/条件分位数和时间／地区重复确认，不能自动叫原创方法。新机制门槛沿用合格覆盖、至少5%CRPS改善、宽度不增、IS下降并有独占消融增量；本轮不会自动开启未查看地区。年度-only需另立年度状态与年内分配合同，不由本轮成功替代。

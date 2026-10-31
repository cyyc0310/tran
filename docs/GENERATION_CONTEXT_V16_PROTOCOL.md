# V16 B支路：历史netG形状的源区粒子试验

2026-10-25，Sol。仅源区pilot，不运行目标或未查看地区，不修改V7–V15。与并行A支路独立；本预算沿用V10的已发布月燃料组成、日历、独立历史netG（假定7或30日滞后），无日/周真实燃料量、无未来G、无天气信息。

**本轮不宣称创新、覆盖改善已确认或新地区晋级。** conditional analog、IPF、支持平滑和有限混合均为标准方法。它用于找可反驳的优化方向；没有新增H2误差模型或校准。所有方法都沿用V14 native的128个粒子及五倍率(.75,1,1.25,1.5,2)，须同时展示倍率变化，不能把扩宽当成模型信息收益。

## 源、基点与时点

七源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS，2025H1逐日181预测；源真值共同有效支持应为1145地区日。每个外层评估地区从proper donor完全排除。其余地区2024-01-25至2024-06-30的完整有效燃料日，使用与V14完全相同的FuelTrajectoryBank、月内归一、等地区/季节/Hellinger核、种子17023和timestamp流、同一V9 harmonic粗组成点与128组成粒子。每一查询重新产生native donor索引、raw CIF和clipped粒子均值，必须逐元素等于V14 R1源native归档，失败即停止。不通过读取旧目标归档来核验。

旧V14 evidence manifest绑定源native和源收据；旧50代码路径与本轮生产/测试/协议三路径构成代码闭包。输入为这些源小时文件、源组成归档、既有独立telemetry及源baseline/元数据。只对注册源区、预测可见时间的netG行转换数值；其他地区先丢弃，不加载其数值。预测阶段只读源2024燃料轨迹，所有28个源预测归档及审计封存后才读取源2025小时标签评分。全部输入/代码与freeze在预测前后、评分前后核验，不覆盖已有或部分输出。

netG是带符号整BA净发电，不等于八正值燃料proxy的G。其历史正值完整日仅用于预测形状，不能把其假定滞后写成真实发布vintage，也不能把IPF目标行总量声称为已知proxy守恒约束。

## 固定历史形状和七方法

预设lags={7,30}日、windows={28,56}日，完整2×2网格。查询日零点t的截止为t−lag。窗口以截止日零点为末端，纳入起点在[cutoff.normalize−window,cutoff.normalize)的完整UTC日，其24个hour_end在(day+1h,…,day+24h)，最后hour_end不晚于截止。只有24小时都有限且严格正值的netG日进入；每个日除以其日均值。优先同星期几历史日，否则使用全部合格日；都没有则所有方法返回native，不使用未来填补。

历史均值形状为选中日的算术均值，再归一到均值1。sample形状在同一池等权抽完整日，用独立固定timestamp流(seed17023+160)。7/30滞后和28/56窗口在看分数前共同冻结，不按源效果改参数。

1. `native`：精确旧V14 B基点。
2. `row_only`：粒子燃料矩阵每小时乘共同正系数。它应保持CIF比例不变（容差1e-9），是必要负对照，不是假改善。
3. `analog_g`：原native donor概率乘固定历史G形状相似度核。donor形状为point mix×donor q的逐小时总量，除其日均值；距离为24小时log形状平方均值，核exp(−distance/.3²)。重新使用相同donor uniform和同一128 coarse draws，是标准conditional analog强对照。
4. `rake_mean_zero`：native相同粒子上用均值G形状，零平滑prior，保留原零support；不可行/不收敛逐粒子回native。
5. `rake_mean_prior`：相同，但固定.001支持平滑。
6. `rake_sample_prior`：相同.001平滑，逐粒子用抽样完整历史日形状。
7. `mixture_mean_prior`：恰64个native偶数索引粒子、64个均值rake奇数索引粒子，保持总128；这是固定分层混合，不对两条CIF路径取平均。

IPF起点a_sf,h=m_draw_sf×q_donor,hf；其日燃料列总量和总能量T均来自该粒子，不来自真实日燃料标签。行目标=shape_h×T/24，燃料列目标=Σ_h a_hf。先以(1−prior)a+prior×列总量/24混合平坦行独立表，再交替缩放行和列；最多500步，最大行/列绝对误差除T≤1e-9才接受。所有失败显式记录并回native；不把一次有限迭代当成功。

.001先验可以在原零单元创建正值，但不能创造原日燃料总量为零的燃料；它不是太阳夜间或可调度性硬约束。零prior独立消融不可省略。若极端组成粒子仅落在该donor当日全零的燃料上，其总能量为0，未定义可rake行量，直接保留native的EF组成回退并标记zero_energy_fallback（未尝试迭代，不把margin误差0当收敛）。analog在point mix下零能量的donor没有可识别G形状，额外G核设为中性。逐粒子记录最终/尝试margin误差、收敛、批次迭代数、新正值单元数及质量。归档保存所有raw粒子、base/analog donor索引、均值/抽样G形状、粗组成点与粒子及center；逐日审计保存实际可用历史起点和cutoff。

## 评价与后续边界

先单独 `--phase predict`，独立review批准且完整seal后再 `--phase score`。评分clip到[0,1000]，finite-ensemble均值为点，倍率在初次clip后围绕其均值展开并再clip；CRPS/小时覆盖/宽度/IS及level/shape辅助指标复用冻结V12。逐日、地区等权macro、最差地区覆盖完整保存，所有28×7×5配置都报告，不只展示赢家。

pilot source结果为开发筛查，同一源标签上的选择不构成独立确认。若某方法看似改善，下一步仍需同预算总误差/hetero/QRF强参考、合法新H2校准和另行冻结确认，不能在此直接查看目标/新地区或宣布最终覆盖成功。没有目标入口，自动新地区触发为false。

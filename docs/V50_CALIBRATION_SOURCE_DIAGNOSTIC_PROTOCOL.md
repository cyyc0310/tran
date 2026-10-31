# V50校准源影响诊断预注册

2026-10-27，Sol。V50正式四阶段与完整零拟合复核已完成后，预先固定此只读诊断；不改V50、不训练模型、不打开目标区真值。

问题：90/30/7天、equal-region/HCP各地区等权小时分位阈值很大或偏保守，是十源普遍中位/尾部风险，还是少数源影响了高阈值？本实验只解释既有校准阈值的来源，不提出新算法，不按诊断结果删源、调lag、改q、重算目标区覆盖或选择未查看地区。

输入限于已封存的70个预测包`predictions/{outer}__{external}.npz`、十个已封存`external_labels/{external}.npz`、冻结的十源键/lag counts与1260条校准记录、completed/prediction/calibration seals、验证PASS回执。程序不打开`outer__outer`预测包、proper/target档案、2025 source fine、PACE/LGEE或未查看数据；0拟合。

固定完整网格7 outer×5base×6method×3lag×2rule=1260。独立重建逐小时分数`max((lo−y)/s,(y−hi)/s,0)`，先证明1260官方q和源日数与已封存calibration.csv相符。每个单源只报本地小时分数ECDF p50/p90/p99、官方q处的局部尾部比例，以及移除该源后其余源重算的equal-region/HCP q及Δq。删除源诊断不会用于产生或评分新的预测区间。

权重按每源总质量相同、源内日与小时相同；经验逆CDF采用最小score使加权ECDF≥预设有限level，整数质量用分母LCM实现，不插值。equal-region有限level=0.9；HCP保留1/(K+1)无穷原子，并使用其有限level，删除源后依新K重算。若要求有限level>1，记录∞与禁止伪造有限q。

输出固定为源码与协议、一个source×configuration长表、摘要：最大官方q重算差、各lag/rule下留一源Δq分布、单源尾部分布和目录级哈希。所有汇总明确为事后描述，不能构成对留出目标独立验证或源标签可交换的证据。若从外十来源于既有开发选择/排除任何一组，必须另立下一轮协议再实验，不能在此文件中改变主结果。

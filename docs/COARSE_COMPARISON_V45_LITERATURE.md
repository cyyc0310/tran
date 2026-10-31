# V45粗比较反馈：针对性先例核查

2026-10-27，Sol。本轮只核与V45数学边界直接相关的原始来源，不是完整系统综述，也不构成新颖性认证。

- Balagopalan等，[Efficient contextual partial monitoring with general function approximation](https://arxiv.org/html/2610.19804v1)，2026-10-14预印本。读摘要、§1的任务设置、§2.2可观测性定义/Assumption2.2及§3相关保证，未通读附录。论文在有限loss/feedback矩阵下处理有上下文的部分反馈，主假设含realizability、global observability与non-degeneracy。我们在同一g下改变未知EF便可改变相对loss，而各预测动作的粗反馈仍相同；一般不能直接写出其loss-difference witness。本文的regret界不能据此移植到当前未识别区间，做一个contextual gate也不是新贡献。该文预印本身份保留，不称已获顶会接收。
- Kirschner、Lattimore、Krause，[Linear Partial Monitoring for Sequential Decision Making](https://jmlr.org/papers/v24/22-1248.html)，JMLR 24(346)，2023。本轮只读官方元数据和摘要；线性partial monitoring、IDS及contextual/kernelized扩展已有先例，不声称已核全部定理条件。
- Liu、de Paula、Tamer，[Prediction Sets and Conformal Inference with Interval Outcomes](https://arxiv.org/html/2501.10117v4)，2026-02-23修订版。读摘要和§4的区间score、Theorem4.1及局部有效性说明，未通读全部证明。区间结果的部分识别、预测与模型不确定性以及共形集合已有方法；Theorem4.1明确使用i.i.d./可弱化为交换性条件，其区间score需要覆盖观测区间。当前重复源月、跨地区机制变化和小时与月度响应差异不能被这条定理自动覆盖。

V45已有sharp界和识别率盘点不能单独主张方法原创。值得继续检验的是：延迟粗信息具体允许更新哪个风险方向，哪些内部水平/小时形状风险必须保留，以及哪种最小本地信息能改变不可识别部分。若设计区间回归、min-max或共形更新，必须保留这些普通强参照，并明确依赖的跨域/时序假设。

这里关于当前任务不具一般global observability的判断，是基于“同粗反馈、不同隐藏EF能产生不同loss差”的推断，不是把有限PM的hopeless分类定理未经约化就当成当前连续电力系统的现成定理。新增可测字段或结构假设可能改变该结论，需另行证明。

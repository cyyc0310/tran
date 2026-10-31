# V184模型准备：九基线有限smoke通过，尚无性能评分

2026-10-29，Sol。九种普通月度分布基线helper已实现，仅完成fold0的short/long_fixed/long_all三预算、首尾两个query检查。没有执行V184全六折粗评分，没有小时新评分或原创性结论。

171/171/597个训练月的三组模型成功拟合，两个query的四个base方法共8个law（支持与概率）在short/long_fixed之间bit-exact。两个query×三预算×九方法共54个未来月量干预law检查通过。long_fixed与long_all共享数据容器，每次扰动均在finally恢复，未污染其他检查。模型的全拟合重建、独立叶概率/OOF残差审计及全量评分仍待完成。

初版smoke r1把“上一年同月”实现成“任一历史年最近同月”，与设计不一致。审查后在任何正式性能评分前改为严格上一年，否则持平；原r1及原源码保留并标记退役，不能再声称其输入hash与当前helper一致。r2由可重复runner执行，绑定完整context数据输入hash并在结束复验。没有依据性能反馈调参。

本项证明有限实现与输入隔离，不证明预测价值。下一应冻结正式runner/协议、补全预检，再执行同预算信息价值实验；不能把V183新增粗数据、8个负控制或54项小范围检查写成论文突破。

有效smoke：`results/v184-monthly-model-smoke-r2`；退役来源：`results/v184-monthly-model-smoke-source-r1`。输入准备另见[V184上下文预检](V184_MONTHLY_CONTEXT_PREFLIGHT_RESULTS.md)。

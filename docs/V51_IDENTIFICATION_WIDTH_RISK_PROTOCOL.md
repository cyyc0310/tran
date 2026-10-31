# V51 月度预测诱导歧义宽度风险特征实验协议

2026-10-27，探索性协议。特征候选在H1正式计算前提出；结果与已有月度-only窗部分重复开发，因此不称独立预注册或确认实验。H1结果查看后，风险特征定义、模型形式及H2时间窗未改变；所有结论均保留事后探索等级。

## 问题及特征

已有同一严格月度-only点预测器只用两个当地份额（热组占比、风光占比）做异方差条件尺度。候选新增“预测诱导热组重排歧义”特征：取目标区合法滞后五组月量`M`、留目标源的上一年同月五组小时profile得到预测小时总量`Ghat_h`及donor热组EF`e_hat`。在假定目标月总量等于这些滞后组量的代理合同下，热量`P_h`只受`0≤P_h≤Ghat_h`及`ΣP_h=M_thermal`约束，故小时热组CIF投影宽为

`w_h = e_hat × [min(Ghat_h,M_thermal) − max(0,M_thermal − (M_total − Ghat_h))] / Ghat_h`。

风险头用整月`w_h`的均值与p90，加回原有热占比、风光占比，形成4特征输入。源平衡Ridge(alpha=10)预测校准区间q90的对数；校准源留一OOF预测给标准化分母，最终用source-equal pooled normalized q90校准。另比较OOF每源q90最大值作保守校准。点预测、小时truth、类别口径、时间窗口、来源权重、头部和阈值与原月度only脚本一致。

## 强参考

同点预测比较：

1. `global_equal_source_q90`：来源平衡pooled绝对残差q90。
2. `conditional_mix_only`：原两特征月组成风险尺度（结果从已完成的`results/v51-monthly-only-risk-h1/h2`读取配对）。
3. `conditional_identification_width`：新增宽度特征，pooled normalized q90校准。
4. `conditional_identification_width_max_source`：同一宽度风险尺度，以校准源最大normalized q90校准。

前推窗为2024H1校准→2025H1评分、2024H2→2025H2评分，逐外层留源；无目标源小时信息进入点头或风险训练。严格门槛仍为宏覆盖90–95%、最差源覆盖≥85%、且相对强同预算基线改善区间效率。不满足则不晋级，不开未查看地区。

## 限制

新宽度只是建立在“滞后月份可代理目标月组成、donor EF正确、donor小时形状可定义G”的**预测诱导宽度特征**，不是当前月份真值约束的sharp物理识别界，也不保证包含未来truth。所有来源已反复用于开发，美国EIA固定EF代理，不是中国可得性或泛化证据。由此得到的最大结论只能是源区风险相关性或一个待新域确认的候选。

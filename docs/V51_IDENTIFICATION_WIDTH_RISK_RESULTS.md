# V51：预测诱导月内热组重排宽度改善宏区间分数，但未修复最差地区

按[协议](V51_IDENTIFICATION_WIDTH_RISK_PROTOCOL.md)在不改点预测器的情况下，加入月度热组总量与donor预测小时总量导出的歧义宽度均值/p90。比较原 pooled q90、已有两特征 mix-only 风险头、新四特征风险头和同头max-source归一化q90。H1/H2均为反复开发的七个美国代理区，不能作为独立确认。

## 宏观配对结果

| 窗口 | 风险/校准 | 宏覆盖 | 平均宽 | 区间分数 | 最差来源覆盖 |
| --- | --- | ---: | ---: | ---: | ---: |
| H1 | pooled q90 | 86.97% | 278.04 | 430.10 | 26.57% MISO |
| H1 | mix-only条件尺度 | 88.75% | 271.30 | 394.51 | 32.28% MISO |
| H1 | 歧义宽度条件、pooled norm q90 | 91.83% | 275.74 | 335.11 | 56.66% MISO |
| H1 | 歧义宽度条件、max-source norm q90 | 96.45% | 363.66 | 384.15 | 77.20% MISO |
| H2 | pooled q90 | 83.19% | 275.63 | 491.86 | 11.97% MISO |
| H2 | mix-only条件尺度 | 85.14% | 270.48 | 459.79 | 15.32% MISO |
| H2 | 歧义宽度条件、pooled norm q90 | 93.38% | 322.69 | 355.26 | 75.21% FPL |
| H2 | 歧义宽度条件、max-source norm q90 | 97.65% | 494.46 | 506.79 | 87.24% FPL |

两个半年中，新尺度都比pooled和mix-only头降低宏区间分数，验证了“预测诱导月内热组可重排范围”含有风险排序信号。H1下新头仅增加约1.6%宽度，区间分数降低15.1%；H2覆盖升至93.4%，但平均宽度也增加17.1%。这是可保留的算法候选增量，不是稳健突破。

它没有通过预定可靠性门槛：pooled norm校准H1/H2最差覆盖分别56.7%和75.2%；换max-source也只有77.2%/87.2%，同时明显加宽，H1宏覆盖96.5%已过覆盖，H2区间分数反而比 pooled norm 头差。最差来源从MISO换成FPL，说明source-specific不交换的剩余风险仍未解决。所有输入是预测时可得的lagged mix和源profile，但宽度本身只是模型诱导特征，不是保证覆盖truth的识别界。

### 支持域门控诊断

事后用训练源4维标准化特征的轴对齐包络定义支持域：包络内用pooled normalized q90，包络外用max-source normalized q90。H1/H2宏覆盖为95.40%/97.28%，宽313/451，区间分数338/464；H1最差MISO仍77.20%，H2最差FPL 87.24%。该规则并未认证校准有效：BPAT在两个半年都频繁落在特征包络外但覆盖很好；H2 MISO也完全域外而覆盖100%，相反H1 MISO域外且严重失覆盖。简单包络不能区分条件迁移好坏，只会对部分域外样本加宽。该门控是post-hoc诊断，不是新增算法，完整复算见`results/v51-identification-width-risk-support-gate/`。

因此判断为：**新增物理语义风险特征在旧源上产生了跨半年宏观区间效率增量，值得保留为论文候选；尚不够支撑可靠小时区间算法论文。**下一轮应只围绕FPL/MISO失效检验是否由特征域外/热组EF内部混合造成，并以固定扩宽、max-source、区域校准/拒绝规则作强对照。不得在七个已开发区反复选定后称确认；若存在明确增量，先冻结，再打开未查看源。

产物：`scripts/v51_identification_width_risk.py`、`scripts/analyze_v51_width_risk_support_gate.py`；风险头结果目录为`results/v51-identification-width-risk-r2-h1/`、`results/v51-identification-width-risk-r2-h2/`，包络诊断输出为`results/v51-identification-width-risk-r3-h1/`、`...-r3-h2/`、`results/v51-identification-width-risk-support-gate/`。旧原始H1/H2完整预测/指标见`results/v51-identification-width-risk-h1/`和`...-risk-h2/`。点MAE数值与base逐项完全一致，风险臂没有偷换中心。

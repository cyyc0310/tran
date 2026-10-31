# V184输入准备：粗监督池扩大，尚未拟合预测模型

2026-10-29，Sol。只完成月度数据容器、训练名单和特征检查，没有拟合V184的九种分布基线，没有计算新的粗/小时性能。后续模型预检和完整评分仍待实施。

ExtendedMonthlyInformation只读取原margin/index和V183的`monthly_history.csv`，将661个历史记录追加到原936个id之后。原M、shares及四个身份字段逐项保持，历史与原(region,month)无重叠；不暴露offset/stop或raw。最近合法记录按日期选择，不因追加id较大而将旧月误作最新月。

| fold | short / long_fixed粗训练标签 | long_all合法粗标签 |
|---|---:|---:|
| 0 | 171 | 597 |
| 1 | 161 | 592 |
| 2 | 151 | 585 |
| 3 | 151 | 588 |
| 4 | 160 | 626 |
| 5 | 170 | 628 |

964组原source训练上下文的base13与ΔS在short/long_fixed之间精确相等，241个test query的base13同样精确相等。每个训练上下文的proper角色、年份、prefix发布时点及不含自身结果均核验。12个query×2容器共24次未来M扰动后，base/history/prefix保持不变。这是特征隔离检查，不能写成24个模型或分布预测干预。

这证明更多合法粗训练上下文可在不增加小时训练输入的条件下准备出来，尚不证明能改善预测。必须按[V184设计](V184_COARSE_HISTORY_VALUE_DESIGN.md)让所有强基线获得相同新增信息，验证base负控制后再评分；粗结果也不能替代原小时gate或未查看地区确认。

归档：`results/v184-monthly-context-preflight-r1`。数值模型、小时传播及原创性均未验证。数据仍是回顾性canonical proxy和合成发布合同。

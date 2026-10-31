# V21：普通native-offset残差控制没有通过实用增量门槛

2026-10-26，Sol。按[冻结协议](NATIVE_REFIT_V21_PROTOCOL.md)完成prepare52644、predict85113、score28067，全部实际exit0；完整七档预测封存后才读取2025源fine评分。[独立实际审计](../results/native-refit-v21/numerical-audit.md)第二次完整执行4748 exit0/PASS，首98561因审计器布尔列比较错误exit1，原脚本与日志保留。本轮为源自适应开发，不是独立确认、新风险机制或未知区测试。

## 所检验的同一任务

所有九输入统一拟合`y−native_mean`，H1完整有效源1100地区日，在每个outer排除后使用其它六教师；教师native bank同时排除outer和教师。固定ET150/leaf10/seed17023、地区等权Scaler与权重。比较native mean24/std24/calendar6，四个packet-only与四个joint(native+packet)输入。packet-only仍以native为offset，不能叫完全无native的信息基线。

查询保持原native128粒子及IDs，只平移中心；三种投影、α=.5/1、共享α0，共55raw arms与275factor曲线，共同1145源地区日。没有重拟合风险、hetero/QRF或粗误差律；clip带来的变窄另行记录，不归为新不确定性机制。H1教师是允许同日/未来H1供体的离线地理OOF，不是当时可发布预测；教师五源/查询六源差异仍在。

## 结果：门槛失败，不能把回退项当赢家

覆盖合格要求宏覆盖90–95%、最差地区≥85%。九个新输入中，仅packet_only_D8有3/30配置合格（day/block/full、α.5、factor1.25），其余八输入全部0/30；native有1/5合格。D8三个合格配置的CRPS26.746835/26.803695/26.910029均比native26.694362更高。D8 day项的宽175.901671、IS200.693002略好，但没有联合性能增量。

| 配置 | CRPS | 宏覆盖% | 最差覆盖% | 宽度 | IS | 覆盖合格 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| native，f1.25 | 26.694362 | 92.9545 | 85.2273 | 176.272555 | 202.247558 | 是 |
| packet_only_D8，day/.5/f1.25 | 26.746835 | 93.3121 | 86.5294 | 175.901671 | 200.693002 | 是 |
| joint_T8，day/.5/f1.25，匹配诊断 | 26.691208 | 92.603 | 83.0492 | 175.867 | 203.077915 | 否 |
| joint_T8，block/.5/f1.5，源选fallback | 27.393631 | 96.0635 | 85.6534 | 210.552117 | 226.065190 | 否 |

T8 fallback宏覆盖超过95%，尽管最差地区超过85%，仍不合格；CRPS与宽度较native分别恶化约2.62%和19.45%。`source_budget_gate=false`。不能事后用不管覆盖的最低CRPS、换factor或把fallback重命名为winner挽救此轮。

## 负结果的准确边界

Native特征并非毫无价值。在完全匹配的30个配置中，joint相对packet-only的宏CRPS在D2为30/30更低、T8为30/30、G5为29/30、D8仅5/30。固定day/.5/f1.25，joint−packet的CRPS差为−.552001/+.049687/−.475954/−.302782；这些重复配置高度相关，不是30次独立检验。T8的匹配改善仍有地区异质性，而且BPAT覆盖83.0492%未过门槛。小点误差或CRPS改善没有变成覆盖合格的整体收益。

因此只能否定**这一固定native-offset ET家族在当前网格的实用增量主张**，不能证明T8或native特征信息论上无价值。也不能把本轮负结果直接当成原V17/V18强中心已被完整检验：原模型拟合`y−daily_anchor`并先clip绝对期望中心，而V21拟合`y−native_mean`、残差不预clip。有限树模型不对可变响应offset保持不变；跨版还存在中心clip差异，不能单因果归因。

原V18 T8 day/.5/f1.25仍为CRPS24.931876、最差覆盖85.4877%、IS196.203187；matched G5为25.057603。T8对G5仅约0.50%，且宽微增，原信息门槛同样失败。这些旧强参考保留，没有用较弱的新残差任务替代它们。若继续检验“原强中心能否更好利用native信息”，下一轮应在同anchor offset下比较packet与joint，并先复现旧packet-only中心/路径；不同时改风险律。详见[逐输入归因和最小后续控制](../results/native-refit-v21/diagnostic-options.md)。

## 独立实际审计与失败记录

69代码/158输入及R1/V18/V17父绑定、七compact NPZ+七JSON、共同H1/2025mask、所有features/response/as-of/权重/Scaler均核验。全部63个固定ET由独立脚本重拟合，保存的δ精确一致，最大差0。214072320个arm raw单元、314875日行、1925地区、275宏、1260匹配对照及2415背景参考核对通过；3850个事先固定Jan/Apr评分行另用显式pair CRPS核验，最大差5.68e−14，完整日表最大差4.55e−13。source gate亦独立重算为false。

首实际审计在主要数值检查后，由于把`native_input_available`布尔列误当数值相减而退出。修复仅将布尔列改为精确身份比较，加混合类型回归，并通过独立通用复审；没有放宽数值容差、修改producer、归档、输出或父字节。第二次重跑了全部63森林和所有评分，没有从失败点跳过核验。

原审计脚本`3799d72a549538a64f8bfe0a28f16f4ffcc2c7030c16a1d68423cad6d4d290ac`保存在`audit_outputs_attempt1.py`；原`audit.log`与副本`audit_attempt1.log`均为`d89e1e3a9ccff21525ba556473b67f44b7e490c2d6b8ee877dc818c79b1b08e3`。修订审计脚本`86a32a2cd6846bc8265c0d9c784dc8b4a57ba24c182d3bf7fdce2586545671cd`；第二日志`03904061aee10f34b6d63d1488aba36fd5c86301fc0dc9671471129cc1794efa`；最终报告`3bea0b9a7120fd106cb16fa4724ca1af420697d0daf50fe4d5e845a0f78b0cbb`。

没有新增目标/未知地区读取，没有H2 fine训练。当前进展是补齐普通控制与识别失败来源，尚不足以宣称原创方法突破、可靠的目标覆盖或实际中国数据通道成立。原目标继续。

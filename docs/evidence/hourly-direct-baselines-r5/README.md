# R5同源标签强基线证据

对应[R5结果](../../HOURLY_DIRECT_BASELINES_RESULTS.md)和[预测前协议](../../HOURLY_DIRECT_BASELINES_PROTOCOL.md)。只含已见US/GB/DE的开发比较，没有未知国家确认。静态装机代理、修订月报模拟t−2和固定因子发电侧代理标签的限制均保持。

`manifest.json`绑定本包各文件SHA及原始文件路径/摘要。`predictions.csv.gz`解压后精确还原全部116064行预测；配合[原2400行带标签队列](../weather-transfer-seasonal-r1/cohort_with_labels.csv.gz)，按region、valid_at、label_row绑定即可复算全部点/形状/中位数/WIS/覆盖/宽度/区间分数。原成员的`row`是该队列的行位置。

`paired_members.tar.gz`包含54个配对银行成员数组，`source_oof.tar.gz`保存48套直接模型的OOF行位置、响应、留块预测及国家层级权重。`source_training_cohort.csv.gz`无损保存35448行源公共输入，OOF的源row对应其行位置；OOF响应是源监督，不可当成部署输入。完整48套拟合模型和6银行仍在本地结果目录，由model_seal绑定，本包不重复存放大型参数。

`results/`保存所有汇总、126条源校准记录、执行计划、运行版本及三类封存。`independent_review/`保存独立Sol复算、来源/OOF/单位/校准及模型audit凭证；不重新训练，也不使用目标标签选型。`diagnostics/`是评分后开发诊断，不是训练特征。

诊断表`actual_thermal`表示目标日逐小时火电占比的算术均值，`public_thermal`是可见历史月电量比例；前者不是日能量比例`sum(thermal_generation)/sum(total_generation)`。小时平均与能量加权口径应分别使用，不据这张诊断表强制匹配真实未来日能量。

旧R4完整源年参考使用额外小时标签，在预测表内标为`extra_full_year_reference`；不可将它当作同35448标签预算主比较。其成员和此前源气候材料分别见[R4证据](../resource-daily-clean-r4/README.md)及[源天气扩展证据](../resource-response-development-r2-r3/README.md)。本包支持独立评分和输入/模型审计，不承诺单靠本包即可重采所有原始能源/GFS或从零重训。

天气重组主结果MAE80.76、WIS38.69、覆盖88.16%、宽300.49；DE→CISO覆盖仅45.83%，US→GB68.06%。直接火电占比在CISO更好却在BPAT严重失效。不能把宏均值收益改写为可靠覆盖或原创性成立。

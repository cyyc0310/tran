# 第三个国家：德国SMARD原始发电数据接入

2026-10-31。此项是源数据接入，不是德国预测成绩，也不代表三国训练集已经口径统一。

## 来源和边界

[SMARD官方数据使用说明](https://www.smard.de/en/datennutzung)说明市场图表数据按CC BY 4.0提供，署名为“Bundesnetzagentur | SMARD.de”。本次仅采集发电数据，不涉及价格。官方说明数据经Bundesnetzagentur核验、来源于ENTSO-E；原始序列仍可能估算和后续修订。

[官方用户手册](https://www.smard.de/resource/blob/205652/63fcff2c9813096fa2229d769da164ef/smard-user-guide-09-2021-data.pdf)定义实际发电为净发电，针对公共供电网络，不含工业封闭网络等部分；原始15分钟量转为MWh。这与中国NBS规上工业发电范围不同，必须保留范围差异。

API直接访问`www.smard.de/app/chart_data/{filter}/DE/index_hour.json`及其周序列，不通过第三方加工数据。API编码参考用于定位接口，研究字段含义以官方图表和手册为准。保留12类原始量：褐煤、硬煤、气、核、海风、陆风、水电、光伏、生物质、其他常规、其他可再生、抽蓄。不把其他可再生静默并入火电，不把抽蓄当一次能源。

## 实际验证与采集

- 已连通官方索引，并下载2022年开头两个周块、12类共24个JSON，探测无请求错误。周块包含边界相邻年份，因此不能把24×168直接称作完整2022小时样本。
- 单位探针：时间戳1641164400000的褐煤小时量3676.5，对应四个15分钟量909.25、913、912、942.25之和恰为3676.5，验证小时接口为聚合电量。原始探针保存在`results/smard-unit-probe.json`。
- 全年2022采集已完成636个周类别JSON，请求错误0；逐文件SHA核验通过，原始缓存位于`data/smard-de/raw/`，终态清单为`results/smard-de-2022-full.json`。按Europe/Berlin的2022历年筛选，12类各有8,760个唯一小时时标，无冲突重复、非有限值或负值。
- 原始年度数据已由下述原始层接入；统一燃料映射和德国目标评分尚未完成。当前不进入任何模型、选型或中国精度结论。紧凑[采集清单](evidence/smard-de-2022/smard-de-2022-full.json)、[质量库存](evidence/smard-de-2022/smard-de-2022-audit.json)和[单位探针](evidence/smard-de-2022/smard-unit-probe.json)已归档。

执行更新：2023年648个周类别文件已下载，请求错误0。新增`transcif.data.smard.load_native`逐记录验证哈希、检查冲突、保留12类，并生成`results/smard-de-native-2022-2023/native_hourly.csv`。两年各8,760小时、全部类别无非有限值或负值；17,520个UTC时标与按Berlin日历生成的完整小时网格逐项相同。边界周跨年文件去重后共1,272个原始文件。仍未生成CIF标签或强制合并其他可再生/抽蓄。[原始层审计](evidence/smard-de-native-2022-2023/audit.json)与[2023采集清单](evidence/smard-de-native-2022-2023/smard-de-2023-full.json)可核验。

收集器为`scripts/fetch_smard_generation.py`，四并发、每请求15秒超时、两次尝试、原文件缓存、逐URL哈希与失败清单。新版本验证结构后才写缓存，坏缓存按内容SHA隔离并重取；index解析/结构失败也记录。新增两项mock网络回归测试通过，独立Python审查确认错误路径修复。

本次2022全年任务在修复前启动，使用旧进程内代码；对应原源码保存为`results/smard-de-2022-full-collector.py`。不将新版本的恢复能力倒写成旧运行事实。2023采集使用修复版完成，终态清单为`results/smard-de-2023-full.json`，启动源码及哈希已自动保存。原始数据、完整缓存继续留本地，紧凑清单随仓库归档。

后续完成字段映射和模型输入合同；以源内验证确定方法后，才按冻结协议安排外层国家测试。上述质量库存是对原始数值的基础检查，不等于核算边界已认证；下载数据不等于已取得独立CO2排放标签。

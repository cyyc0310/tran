# TransCIF v0.2 清理与重构（2026-10-22）

对象是 `/Users/cyyc0310/Desktop/tran`，起点 `f37a99c0a2b85a86df3cb07b34ce2487386bcc89`，起点工作树干净。工作区的 `tran-stable` 是 v0.1 证据归档，AI4S 当前仍固定旧版接入；本轮没有修改它们。原始数据、模型和历史实验均保留。

## 删除依据

- 删除 `physics/astro.py` 中四个未进入任何当前模型、数据入口或基线的太阳几何/晴空特征函数；仓库外业务适配也未引用。活跃的 NumPy 风功率曲线迁至 `physics/wind.py`，保留其契约测试。删除的 9 项测试仅验证被删除功能。
- 删除 `calendar_features(..., tz_offset=...)` 的旧兼容分支。真实入口一直传 manifest 的 IANA 时区；测试改用 `Etc/GMT+8` 验证固定偏移，现有夏令时测试保留。
- 删除无调用方的 `ModelInputs.to`。`stack_inputs(..., device=...)` 仍是训练/推理构造设备张量的入口。
- 删除 `build/` 中 33 份生成的重复源码。删除前逐文件验证与起点 Git 的 `src/` 内容完全一致，可恢复路径及哈希另存证据。
- 其余公开数据适配、模型消融、机制集合、基线和约束诊断都有活跃调用或明确研究用途，未因结果不利而删除。

## 重构与文档更正

- `inference.predict` 独立于训练；物理点预测复用已经构造的公共输入，避免重复校验/堆叠。
- 源验证评分集中到 `evaluation.metrics.source_validation_metrics`，明确保留时间选模的 pooled CRPS 与等区域点误差、留源组选模的全指标区域宏平均。
- 岭回归的目标预测只计算一次；两种树基线的同构拟合/残差情景编排合并，输出字段不变。
- README 移除重复的开发叙述；更正“尚无公开预报接入器”和“仅两个残差基线”等过时表述。`RESEARCH` 负责假设和判据，`RESULTS` 负责已核验结果；不把计划写成已实现或已证明。
- 新增结果证据索引，对 12 套人工开发 benchmark、144 次历史运行逐项复核输入/manifest、历史 Git 源码指纹和缓存预测评分。四组配对区间保留为归档探索性分析，没有伪装成新增盲测。

内部 Python 导入迁移：`training.fit.predict` → `inference.predict`，`physics.astro.wind_capacity_factor` → `physics.wind.wind_capacity_factor`。仓库调用均使用新入口，不新增兼容包装；CLI、checkpoint 配置/权重格式和预测产物字段保持；已用重构前的单模型和机制集合 checkpoint 实际运行新 CLI。`fit.predict` 因训练内部导入仍可解析；依赖已删除 `astro`、`tz_offset` 或 `ModelInputs.to` 的外部脚本需迁移。历史源码哈希按其原 Git 版本核对。

## 验证

使用工作区已有的 Python 3.12 环境，以及此前记录来源的本地 ecCodes 依赖路径；桌面仓库原有虚拟环境未重建。完整天气解码测试实际执行，无跳过。

| 阶段 | 通过 | 失败/跳过 |
| --- | ---: | ---: |
| 原始基线 | 195 | 0 / 0 |
| 删除未用太阳特征及对应测试 | 186 | 0 / 0 |
| 移除固定偏移兼容分支 | 186 | 0 / 0 |
| 完成重构，新增评分权重协议回归 | 187 | 0 / 0 |

在隔离临时目录中导出起点 Git 源码作为基线，完成后删除该临时副本；核对基线运行记录的源码指纹逐项匹配起点 Git。固定人工数据 seed=19、训练 seed=8、3 步、4 路径，分别比较 point（含两种树）、CRPS、joint（含机制集合）三条流程。重构前后所有预测字段、模型/集合权重、训练历史、机制选择与评分**逐值完全一致**。这是行为等价检查，不是预测质量实验。

证据在 `results/refactor-validation-20261022/`：四份 JUnit、`before-original/`、`after-final/`、`equivalence-final.json`；`deleted-build-files.json` 为生成副本删除清单。早期验证还保留在 `before/`、`after/`、`equivalence.json`，以隔离 Git 基线的最终对照为准。结果评分/输入及历史源码核对见 `docs/evidence/research-results-20261022.json`。源码 Python 总行数 3056→3011，测试 1990→1944；这两个数量不包含文档或本地实验产物。

仍有两条原有 pandas FutureWarning；没有屏蔽测试。真实数据基准、核算模型扩展和论文创新证据是后续研究工作，不把本次代码清理称为已解决。

独立复审：GPT-6 Astra / ultra 对本轮代码给出 APPROVE，Python 专项另运行 9 项定向测试通过；未发现新增阻断问题。完整科学结论及 P0/P1/P2 补证据判据见 [独立评审](PAPER_REVIEW_ASTRA.md)，代码通过不等于论文投稿证据充分。

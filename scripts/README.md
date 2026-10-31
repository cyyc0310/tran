# 实验与维护脚本

脚本维持原位置，因为协议、输入清单和父实验收据引用了路径及源码哈希。编号表示研究过程，不表示所有版本均可用，也不保证每个编号只对应一个研究分支。不要按最大编号自动选择方法。

| 类型 | 入口与用途 |
|---|---|
| 基础模型 | `run_experiment.py`、`run_benchmark.py`、`predict.py` |
| 数据采集/适配 | `collect_*.py`、`stage_*.py`、NBS/EIA/天气相关脚本；先读数据协议 |
| 三关与初期迁移 | `run_research_gates.py`、`run_*_v*.py`，实现通常在 `src/transcif/research/` |
| 风险识别与信息价值 | `vNNN_*.py`；每项对应 `docs/` 下协议/结果/更正 |
| 官方NBS数据 | `v394_nbs_all_province_monthly_snapshot.py`、`analyze_v394_all_province_lopo.py` |
| 年度官方因子对照 | `v404_nested_region_conformal_mix_intervals.py` 至 `v407_temporal_forward_shrinkage_confirmation.py` |
| 中国模拟压力复测 | `v410_china_simulated_cif_scenario_stress.py`、`v411_level_shape_scenario_exploratory.py`；均为已查看情景上的开发分析 |
| 仓库维护 | `maintenance_snapshot.py`；复制小型摘要并生成脚本/文档/本地产物库存 |

逐文件说明和 SHA-256 见 [脚本目录](../docs/catalog/scripts.csv)。核心包导入依赖 `pip install -e '.[research]'`，个别归档脚本还依赖原始输入、父实验、人工固定时滞或绝对路径；读取其协议后再运行。不要将测试中构造的数据、修订后历史月表或模型情景数据描述成真实发布档案。

创建新的紧凑快照（输出目录须不存在）：

```bash
python scripts/maintenance_snapshot.py --output docs/evidence/research-snapshot-YYYYMMDD
```

该操作不修改 `results/`，只复制固定文件名、大小不超过512 KiB的摘要/宏观指标并计算哈希。所有旧摘要都会保留其原状态；失败或无效结果不会自动过滤。大表、checkpoint、原始数据留在 Git 忽略目录。

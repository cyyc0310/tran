# TransCIF

研究如何用其他国家的小时运行数据训练模型，在中国等没有本地小时 CIF 历史的地区，利用已公开的月度/年度统计及可获得的外生信息，预测未来小时碳强度。目标是可复用的跨域训练、零目标小时标签部署和评价框架。

按用户 2026-10-31 的纠偏，后续优先接通与中国信息条件一致的跨国小时实验，再检验源域训练方式与燃料运行关系的迁移。信息可识别性、最小观测和弃权研究作为辅助诊断；具体顺序见[跨域迁移研究计划](docs/CROSS_DOMAIN_TRANSFER_RESEARCH_PLAN_20261031.md)。

当前中国主输入是国家统计局省级火电、水电、核电、风电、太阳能月度发电量。火电合计不能视作煤/气/油细分；规上工业发电不等于全口径供电；本地发电侧强度与含跨省受电的消费侧因子分别评价。

**截至 2026-10-31：尚未证实具有稳定覆盖与区间效率优势的小时 CIF 迁移算法。** 已形成公开字段审计、小时风险部分识别求解器、信息预算比较、源外/时间外实验和大量负结果。中国情景小时数据是模型投影，EIA 固定因子标签是代理，都不等于实测中国小时 CIF。

- [当前研究重点：国外训练、中国无小时标签迁移](docs/CROSS_DOMAIN_TRANSFER_RESEARCH_PLAN_20261031.md)
- [近四日成果与当时的研究判断（10月28–31日）](docs/reports/2026-10-28_31.md)
- [文档入口与结果优先级](docs/INDEX.md)
- [实验脚本分类与运行约束](scripts/README.md)
- [可随 Git 获取的结果快照](docs/evidence/research-snapshot-20261031/README.md)
- [完整研究过程记录](docs/BREAKTHROUGH_SEARCH.md)

## 仓库结构

| 路径 | 用途 |
|---|---|
| `src/transcif/` | 调度模型、公共数据合同、训练、推理与评分 |
| `src/transcif/research/` | 有限信息下的组成预测、重组、风险迁移实验模块 |
| `scripts/` | 数据采集、实验 runner、审计与维护；原路径保留以兼容历史哈希 |
| `tests/` | 核心和研究模块的既有回归测试 |
| `docs/` | 协议、结果、更正、文献与审查记录 |
| `docs/catalog/` | 文档/脚本目录、本地大产物库存 |
| `docs/evidence/research-snapshot-20261031/` | 原样复制的紧凑结果摘要、CSV 和 SHA-256 清单 |
| `data/`, `results/`, `external/` | 本地数据、大实验结果和外部资料；不进入 Git |

## 安装与基础运行

Python 3.10+，建议使用独立虚拟环境：

```bash
python -m pip install -e '.[dev,research]'
python -m pytest -q
python scripts/run_experiment.py --demo --output results/demo-new
```

人工 demo 用于软件流程验证。真实数据需要符合 [DATA](docs/DATA.md) 的字段、估计对象和发布时间约束。天气 GRIB 工具另需 `.[weather]` 和可用的 ecCodes 动态库。

```bash
python scripts/run_experiment.py --manifest data/public/manifest.json \
  --target TARGET --leave-jurisdiction-out --output results/public-run
python scripts/predict.py --checkpoint results/public-run/model.pt \
  --manifest data/public/manifest.json --target TARGET \
  --origin '2023-03-20T00:00:00Z' --output results/prediction.json
```

没有目标标签时，将 manifest 的 `labels` 设为 `null` 或使用 `--no-target-scoring`。运行研究脚本前阅读同版本协议；很多脚本读取本地归档、父实验文件或固定路径，并不支持仅 clone 后直接运行。入口齐全不代表历史数据和发布版本已经认证。

## 证据与复现

紧凑快照保留原始文件内容和哈希，包含有利、失败、无效和已替代运行；纳入快照不代表结果获得批准。结果优先级以各版本的更正稿和四日成果报告为准。尤其 V404 的早期未脚本化数字已撤回，V410–V411 的情景标签已在 V194–V200 打开，不能称独立未知情景确认。

本地约 88 GB 的结果与原始数据保留，不上传模型、小时预测大表、第三方原始数据或临时缓存。要复现历史实验，需要按快照清单恢复同一输入及脚本版本；当前快照不能代替缺失的原始数据许可和历史发布档案。

原 v0.2 调度模型与三关研究的技术背景见 [RESEARCH](docs/RESEARCH.md)、[三关结果](docs/THREE_GATES_RESULTS.md)、[清理与重构记录](docs/REFACTOR.md)。整理前 README 保存于 [历史入口](docs/archive/README_BEFORE_20261031.md)。

# TransCIF 代码库重构方案（Refactor Plan）

> 目标：把当前 `scripts/` 下 38 个平铺脚本（其中既含被当作库的核心模块，又含各研究方向实验入口、数据下载、出图、诊断脚本）重构为以 `src/transcif/` 为正式 Python 包、目录按 `data / test / zeroshot / <方向>` 归类的清晰结构。
> 作者视角：当前代码"都是强硬的测试脚本"——核心可复用代码（model / pipeline / data / physics）与一次性实验脚本混在一起，无法被当作库复用，也找不到边界。

---

## 一、现状诊断

### 1.1 核心矛盾

`src/transcif/` 下规划了 7 个语义清晰的子包（`models/ data/ physics/ training/ evaluation/ calibration/ config/`），但**全部为空目录**。与此同时，`scripts/` 里 38 个脚本实际承担了所有逻辑：

| 类别 | 文件 | 本质 |
|---|---|---|
| **核心库（被 import）** | `transcif_model.py`、`transcif_data.py`、`transcif_pipeline.py`、`transcif_conformal.py` | 应进 `src/transcif/` 的模型/数据/物理/校准层 |
| **方向模块（被 import）** | `transcif_rag.py`、`transcif_phys_irm.py`、`transcif_causal.py`、`transcif_icl.py`、`transcif_hier.py` | 5 个研究方向的模型代码，应进 `src/transcif/zeroshot/` 或 `research/` |
| **Benchmark / 横向对比** | `run_unified_eval.py`、`run_supervised_baselines*.py`、`carboncast_analysis.py`、`ablation_study.py`、`temporal_ood.py`、`optimize_weak_regions.py` | 跨方法 / 跨 29 区域对比与基线，**专门做测试**，应进 `scripts/benchmark/` |
| **单方向实验入口（CLI）** | `run_{rag,phys_irm,causal,icl,hier}_eval.py`、`probe_*.py` | 单个研究方向的运行/诊断脚本，应进 `scripts/experiments/` |
| **数据下载** | `download_uk_regions.py`、`download_eia930_data.py`、`generate_nemed_regions.py` | 数据准备，应进 `scripts/data/` |
| **出图/分析** | `make_*_figures.py`（×4）、`carboncast_analysis.py`、`verify_paper_numbers.py` | 产物生成，应进 `scripts/figures/` 或 `scripts/analysis/` |
| **定理/验证** | `theorem1_physics_bound.py`、`theorem2_transfer_bound.py`、`validate_us_data.py` | 理论验证，应进 `scripts/verify/` |

### 1.2 关键依赖事实（来自实际 import 图）

- `transcif_pipeline.py` 是"上帝模块"：既含数据加载（`load_region_data`、`build_windows`）、物理重建（`cif_from_shares`）、训练（`train_zero_shot`、`train_patchtst`）、推理（`zs_plus_predict`）、评估（`compute_metrics`、`evaluate_target`），又在文件尾部**反向 import 5 个方向模块**做分发。
- 顶部常量 `SEQ_LEN / HORIZON / DATA_DIR / RESULTS_DIR / TRAIN_STRIDE …` 定义在 `transcif_pipeline.py`，被几乎所有脚本引用——应集中到 `config/`。
- `src/transcif/__init__.py` 不存在，包未正式初始化；`tests/transcif/` 仅存 `.pyc`，源码测试已丢失。
- `pytest.ini` 的 `testpaths = tests`，但目前无 `test_*.py` 源码可跑。

### 1.3 目标读者关心的痛点

1. 无法 `import transcif` 当库用（所有逻辑锁在 `scripts/`）。
2. 跑一个新实验要读懂 38 个平铺文件，找不到边界。
3. `transcif_pipeline.py` 单文件 500+ 行，跨数据/训练/推理/评估/分发，改动风险高。
4. 5 个研究方向与"基线零样本"共用同一套数据/物理层，但当前模块边界模糊。

---

## 二、目标目录结构

以**"可复用库放 `src/`，一次性脚本放 `scripts/`"** 为根本原则。核心库正式落地到 `src/transcif/`，实验脚本按"数据 / 验证 / 出图 / 方向实验"分组。

```
transcif/
├── src/transcif/                  # 正式 Python 包（可 pip install -e .）
│   ├── __init__.py
│   ├── config.py                  # 全局常量 + region config 加载（SEQ_LEN/HORIZON/DATA_DIR/ef_nr…）
│   ├── data/                      # 数据层
│   │   ├── __init__.py
│   │   ├── loaders.py             # load_region_data, discover_uk_regions
│   │   ├── windows.py             # build_windows, cif_from_shares（物理重建也归此或 physics）
│   │   └── dataset.py             # 统一的 Dataset / DataLoader 封装
│   ├── physics/                   # 物理分解层（Theorem 1/2 落地）
│   │   ├── __init__.py
│   │   ├── decompose.py           # cif_from_shares, error bound
│   │   └── bounds.py              # theorem1_physics_bound, theorem2_transfer_bound 核心公式
│   ├── models/                    # 模型层
│   │   ├── __init__.py
│   │   ├── base.py                # AdaptivePersistDLinear, RevIN, ramp_aware_loss, MissingMaskAugmentor
│   │   ├── patchtst.py            # train_patchtst
│   │   └── zeroshot/              # 5 个研究方向作为子包，互不污染
│   │       ├── __init__.py
│   │       ├── base_zs.py         # train_zero_shot, zs_plus_predict, evaluate_target（基线 ZS/ZS+）
│   │       ├── rag.py             # ← transcif_rag
│   │       ├── phys_irm.py        # ← transcif_phys_irm
│   │       ├── causal.py          # ← transcif_causal
│   │       ├── icl.py             # ← transcif_icl
│   │       └── hier.py            # ← transcif_hier
│   ├── calibration/               # 测试时校准（ZS+ / conformal）
│   │   ├── __init__.py
│   │   ├── zs_plus.py             # zs_plus_predict 的校准逻辑（从 pipeline 抽离）
│   │   └── conformal.py           # ← transcif_conformal
│   ├── evaluation/                # 评估指标与 LORO runner
│   │   ├── __init__.py
│   │   ├── metrics.py             # compute_metrics
│   │   └── loro.py                # 29 区域 LORO 实验编排
│   └── training/                  # 训练工具（scheduler 等）
│       ├── __init__.py
│       └── schedulers.py          # get_cosine_warmup_scheduler
│
├── scripts/                       # 一次性 CLI 脚本（不再含库代码）
│   ├── data/                      # 数据下载
│   │   ├── download_uk_regions.py
│   │   ├── download_eia930_data.py
│   │   └── generate_nemed_regions.py
│   ├── verify/                    # 定理 / 数据校验
│   │   ├── theorem1_physics_bound.py
│   │   ├── theorem2_transfer_bound.py
│   │   └── validate_us_data.py
│   ├── figures/                   # 出图
│   │   ├── make_architecture_figures.py
│   │   ├── make_mae_overview.py
│   │   ├── make_paper_figures.py
│   │   └── make_submission_figures.py
│   ├── benchmark/                 # 跨方法 / 跨 29 区域横向对比与基线（专门做测试）
│   │   ├── run_unified_eval.py        # 主 benchmark 编排器：29 区域 × 5 种子 LORO，可开关各方向对比
│   │   ├── run_supervised_baselines.py
│   │   ├── run_supervised_baselines_v2.py
│   │   ├── carboncast_analysis.py      # 与 CarbonCast 外部基线对比
│   │   ├── ablation_study.py           # 消融（横向对比性质）
│   │   ├── temporal_ood.py             # 时间 OOD 泛化基准
│   │   └── optimize_weak_regions.py    # 弱区域专项基准
│   ├── experiments/               # 单方向实验入口（run_*_eval + probe_*）
│   │   ├── run_rag_eval.py
│   │   ├── run_phys_irm_eval.py
│   │   ├── run_causal_eval.py
│   │   ├── run_icl_eval.py
│   │   ├── run_hier_eval.py
│   │   ├── probe_final_multiseed.py
│   │   ├── probe_multibranch_fusion.py
│   │   ├── probe_uk01_diagnosis.py
│   │   ├── deployment_warmup.py
│   │   └── verify_paper_numbers.py
│   └── run_downstream_chain.sh
│
├── tests/                         # pytest 恢复（重建丢失的 test_*.py）
│   ├── data/test_loaders.py
│   ├── physics/test_decompose.py
│   ├── models/test_base.py
│   └── zeroshot/test_base_zs.py
└── pyproject.toml                 # 取代 requirements.txt，声明 package + entry points
```

---

## 三、模块拆分映射（transcif_* → 新包）

| 原文件 | 迁入位置 | 说明 |
|---|---|---|
| `transcif_pipeline.py` | **拆 5 处** | `load_region_data`/`discover_uk_regions` → `data/loaders.py`；`build_windows` → `data/windows.py`；`cif_from_shares` → `physics/decompose.py`；`train_zero_shot`/`zs_plus_predict`/`evaluate_target` → `models/zeroshot/base_zs.py`；`compute_metrics` → `evaluation/metrics.py`；`train_patchtst` → `models/patchtst.py`；`get_cosine_warmup_scheduler` → `training/schedulers.py`。**反向 import 的 5 个方向模块改为在实验入口处显式调用，不在库内做分发。** |
| `transcif_model.py` | `models/base.py` + `models/patchtst.py` | `AdaptivePersistDLinear`/`RevIN`/`ramp_aware_loss`/`MissingMaskAugmentor` → base |
| `transcif_data.py` | `data/` | 数据增强与损失相关 |
| `transcif_conformal.py` | `calibration/conformal.py` | 保形预测 |
| `transcif_rag.py` | `models/zeroshot/rag.py` | 方向一 |
| `transcif_phys_irm.py` | `models/zeroshot/phys_irm.py` | 方向二 |
| `transcif_causal.py` | `models/zeroshot/causal.py` | 方向三 |
| `transcif_icl.py` | `models/zeroshot/icl.py` | 方向四 |
| `transcif_hier.py` | `models/zeroshot/hier.py` | 方向五 |
| 顶部常量 | `config.py` | `SEQ_LEN/HORIZON/DATA_DIR/RESULTS_DIR/TRAIN_STRIDE/TEST_STRIDE/TRAIN_FRACTION` + region config 加载 |

---

## 四、重构原则（防回归）

1. **只搬不重写**：本次重构是"搬迁 + 分层"，不改动算法逻辑。每个 `transcif_*.py` 内的函数体原样移到新模块，仅调整 import 路径。
2. **去掉上帝模块的反向分发**：`transcif_pipeline.py` 尾部的 `from transcif_rag import …` 等分发逻辑删除，改由各 `run_*_eval.py` 直接 `from transcif.models.zeroshot.rag import …`，依赖方向从"库依赖方向"翻转为"方向依赖库"，彻底解耦。
3. **常量集中**：所有脚本里散落的 `SEQ_LEN=336` 字面量统一替换为 `from transcif.config import SEQ_LEN`，消除硬编码漂移。
4. **包可安装**：新增 `pyproject.toml` 声明 `[tool.setuptools.packages.find]` 包含 `src/transcif`，README 的安装步骤从 `pip install -r requirements.txt` 升级为 `pip install -e .`。
5. **保留 `scripts/` 的兼容性**：`scripts/` 下脚本用 `PYTHONPATH=src` 或更优的"可编辑安装"方式运行；迁移期间允许 `scripts/` 内 import 仍指向新包。

---

## 五、渐进式迁移步骤（建议分 5 个 PR）

> 每一步都可独立验证（`pytest` + 跑一个 `run_*` 脚本），不破坏现有实验结果。

**Step 1 — 建包骨架与 config**
- 新建 `src/transcif/__init__.py` 及所有子包 `__init__.py`（空壳）。
- 抽出 `config.py`（常量 + region config 加载器）。
- 验证：`python -c "import transcif"` 成功。

**Step 2 — 迁移数据层与物理层**
- `data/loaders.py`、`data/windows.py`、`physics/decompose.py`、`physics/bounds.py`。
- 把 `theorem1/2_*.py` 的**核心公式**抽到 `physics/bounds.py`，脚本本身留 `scripts/verify/` 作为 CLI 入口。
- 验证：`tests/data/test_loaders.py`（重建）跑通。

**Step 3 — 迁移模型层与基线 ZS**
- `models/base.py`、`models/patchtst.py`、`models/zeroshot/base_zs.py`（基线 train/eval）。
- 验证：跑 `run_unified_eval.py` 结果与迁移前 JSON 一致（已存的 `results/` 作 baseline 对照）。

**Step 4 — 迁移 5 个方向模块到 `models/zeroshot/`**
- rag / phys_irm / causal / icl / hier 各就各位。
- 删除 `transcif_pipeline.py` 尾部分发逻辑，更新 5 个 `run_*_eval.py` 的 import。
- 验证：5 个方向 `run_*_eval.py` 全部跑通，输出与历史日志一致。

**Step 5 — 整理 scripts 子目录 + 恢复 tests + pyproject**
- 把脚本按用途移入 `scripts/{data,verify,figures,benchmark,experiments}/`，其中 benchmark 类（unified / supervised baselines / carboncast / ablation / temporal_ood / optimize_weak）单独成 `scripts/benchmark/`，与单方向 `experiments/` 明确分开。
- 重建 `tests/` 下 4 个基础测试文件（之前 `.pyc` 对应的源码）。
- 新增 `pyproject.toml`，更新 README 安装说明。

---

## 六、风险与注意点

- **`transcif_pipeline.py` 是最大单点**：拆分时务必逐函数核对归属，建议用 `git mv` + 小心编辑，保留 blame 历史。
- **`tests/transcif/*.pyc` 是死产物**：说明历史上有测试源码但被删，重构时应重建而非恢复 `.pyc`。
- **实验结果不可变**：`results/` 下的 JSON/日志是论文证据，重构后必须能复现，故每步都要做"输出一致性"校验。
- **跨方向共享代码**：`ramp_aware_loss`、`RevIN`、`MissingMaskAugmentor` 被多个方向复用，必须留在 `models/base.py`，不能各自复制。

---

## 七、预期收益

1. `import transcif` 当库用，README 的 pipeline 描述与真实代码 1:1 对应。
2. 新同学按 `data → physics → models/zeroshot → evaluation` 顺序即可读懂全栈。
3. 5 个研究方向各自独立子包，互不 import，便于按 `RESEARCH_DIRECTIONS.md` 分头投稿、独立迭代。
4. 测试恢复后，每次重构有回归保护。

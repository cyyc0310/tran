# Cross-Condition Control System Identification Benchmark v1.2.0

- Tag: `control-benchmark-v1.2.0`
- Repository: [`cyyc0310/tran`](https://github.com/cyyc0310/tran)
- Package: `system-identification-benchmark/`
- Release date: 2026-11-03

## Changes from v1.1.0

- Adds a sparse-output evaluation of the Forgione–Piga continuous-time neural state-space hybrid structure on Cascaded Tanks, pinned to upstream commit `956aa7ecd73061b0321c6f6098d74c9e4a7ce67a` under Apache-2.0. This adapts the published structure and main training recipe; it is not a reproduction of the authors' original reported score.
- Adds the training and sparse-prediction protocol, original upstream model/training files and license, three trained checkpoints, per-block predictions, training receipts, result summary, and a verifier that replays predictions without retraining.
- Adds regularized ARX(2,2) source validation and results as a stronger linear comparison, with its own training and verification scripts.
- Preserves all v1.1.0 files and results. The InTaSet author Neural ODE remains unavailable in the checked public sources; the InTaSet NODE remains an independent implementation.

## Cascaded Tanks comparison

On the same single, previously inspected test record, the frozen author-structure hybrid and ARX(2,2) NRMSE are:

| Method | 4 s | 12 s | 20 s |
|---|---:|---:|---:|
| Forgione–Piga author-structure hybrid | 0.02332 ± 0.00019 | 0.03564 ± 0.00015 | 0.04327 ± 0.00035 |
| Regularized ARX(2,2) | 0.02590 | 0.05603 | 0.10550 |
| Python N4SID | 0.04627 | 0.73396 | 16,122.50932 |

The three hybrid seeds measure initialization and optimization variation, not three physical experiments. The fitted latent upper-tank state is negative for about 51% of the source sequence. Output predictions are finite, but some leave the source-record range; that range is not a hardware safety limit. Recorded future pump inputs are used within each prediction block, so the task is offline conditional prediction rather than online forecasting or closed-loop control. These test labels had already been seen during development; this is not blind confirmation.

The results show a strong fit for this fixed hybrid implementation on this one record and protocol. They do not establish a stable winner across transfer directions and systems, prove physical validity of the latent state, reproduce the original paper's score, or validate a new algorithm.

## Reproduce the added comparisons

Use CPython 3.12.13 and install the main dependencies as described in [`README.md`](README.md). Then run:

```bash
python scripts/cascaded_tanks_arx2_regularized_ablation_v1.py
python scripts/verify_cascaded_tanks_arx2_regularized_ablation_v1.py
python scripts/cascaded_tanks_author_hybrid_sparse_benchmark_v1.py \
  --iterations 10000 --seeds 0 1 2
python scripts/verify_cascaded_tanks_author_hybrid_sparse_benchmark_v1.py
```

The author-hybrid verifier checks upstream source and dataset hashes, training receipts and checkpoint hashes, replays 73,428 stored prediction rows without retraining, and recomputes 72 per-seed metrics and 24 summaries. Full protocols, physical diagnostics, and output hashes are in [`CASCADED_TANKS_AUTHOR_HYBRID_SPARSE_RESULTS_20261103.md`](docs/algorithmic-control-v1/CASCADED_TANKS_AUTHOR_HYBRID_SPARSE_RESULTS_20261103.md).

The release candidate was extracted to a clean directory. From that copy, the author-hybrid verifier passed on all 73,428 prediction rows, the regularized ARX verifier replayed 3,066 rows, and the pre-existing Cascaded Tanks source/result and Python N4SID verifiers passed. This confirms the archive paths and receipts work outside the research worktree; it does not create new physical records or blind confirmation. See [`CLEAN_COPY_REPLAY_RECEIPT_v1.2.0.md`](CLEAN_COPY_REPLAY_RECEIPT_v1.2.0.md) and [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json) for the replay details and per-file hashes.

The included 4TU Cascaded Tanks dataset archive and its derived experiment artifacts retain the CC BY-SA 4.0 attribution and terms. The Forgione–Piga source retains its separate Apache-2.0 license. MIT covers only project-authored code and prose; see [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## 中文说明

本版在 v1.1.0 基础上加入 Forgione–Piga 公开连续时间神经状态空间混合结构在 Cascaded Tanks 上的稀疏输出评测，并加入正则 ARX(2,2) 强基线。上游模型固定在 commit `956aa7ecd73061b0321c6f6098d74c9e4a7ce67a`，代码按 Apache-2.0 保留。该实现适配了原方法结构和主要训练配方，但不是论文原始分数的复算。

在一条已查看的测试记录上，作者结构混合模型 4/12/20 秒 NRMSE 为 `0.02332 ± 0.00019 / 0.03564 ± 0.00015 / 0.04327 ± 0.00035`，正则 ARX(2,2) 为 `0.02590 / 0.05603 / 0.10550`。三种子只反映初始化/优化差异，不是三次物理实验。混合模型约 51% 的上水箱潜状态为负；经验数据范围也不是硬件安全边界。所有结果均使用已记录的后续泵输入，属于离线条件预测；测试标签在开发期间已看过，因此不是盲测。

包内包含协议、固定上游模型源码及许可证、训练检查点、逐块输出、收据和独立核验器。核验器无需重训即可重放 73,428 行预测并复算指标。该比较只支持对这条记录和固定协议的观察，不支持普遍算法胜出或新的算法主张。项目代码 MIT 许可不覆盖双水箱数据、派生结果或 Apache-2.0 上游源码。

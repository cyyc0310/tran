# Cross-Condition Control System Identification Benchmark v1.1.0

- Tag: `control-benchmark-v1.1.0`
- Repository: [`cyyc0310/tran`](https://github.com/cyyc0310/tran)
- Package: `system-identification-benchmark/`
- Release date: 2026-11-03

## Changes from v1.0.0

- Adds a two-state Python N4SID comparison for Cascaded Tanks, with its own sparse-observation protocol, pinned upstream source, output files, and independent verifier.
- Keeps the v1.0.0 InTaSet, Cascaded Tanks, and Nano-Drone files and scores unchanged.
- Adds the NFourSID 1.0.2 source under its original MIT license and a separate dependency lock for its plotting dependencies.
- Updates the repository landing pages to identify control-system identification as the current project focus.

## Result and its limits

On the one public Cascaded Tanks test record, the two-state Python N4SID baseline has NRMSE 0.04627, 0.73396, and 16,122.50932 at 4-, 12-, and 20-second horizons. Source-side selection chose three Hankel block rows from eight candidates. The final state matrix has spectral radius 1.02529; at 20 seconds, 45.69% of predictions are negative and 93.92% fall outside the source-record output range.

The target labels had been inspected during development. This is not blind or independent confirmation. The implementation and sparse-observation contract differ from MATLAB `n4sid(data_id, 2)` and from the authors' continuous validation-record simulation; it is not an exact reproduction of their baseline. The result describes this implementation on one record and does not establish that N4SID generally performs poorly. The full protocol, comparisons, and output hashes are in [`CASCADED_TANKS_PYTHON_N4SID_SPARSE_RESULTS_20261103.md`](docs/algorithmic-control-v1/CASCADED_TANKS_PYTHON_N4SID_SPARSE_RESULTS_20261103.md).

These results add a reproducible comparison, not an algorithmic win. They do not establish a stable winner across directions and systems, blind generalization, closed-loop performance, or hardware safety limits.

## Reproduce the added comparison

Use CPython 3.12.13 and first install the main benchmark lock as described in [`README.md`](README.md). Then install the supplemental dependencies and run:

```bash
python3.12 -m pip install -r requirements-cascaded-tanks-n4sid-v1.lock.txt
python3.12 scripts/cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
python3.12 scripts/verify_cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
```

The verifier refits all eight source-side candidates, checks the selected model and source hashes, and independently replays 3,066 target predictions. The source archive is included under CC BY-SA 4.0; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). MIT covers project-authored code and prose only. The vendored NFourSID source retains its separate MIT notice, and MIT does not relicense datasets or result artifacts.

---

## 中文说明

本版在 v1.0.0 基础上新增一项二状态 Python N4SID 对照：包含稀疏观测协议、固定版本的上游源码、结果文件和独立核验器；原有版本的实验数据与分数保持不变。

在一条此前已查看的 Cascaded Tanks 测试记录上，4/12/20 秒 NRMSE 分别为 0.04627/0.73396/16,122.50932。最终状态矩阵谱半径为 1.02529，长跨度预测明显发散。这是开发期比较，不是盲测或独立确认；Python 实现与稀疏预测合同不同于作者 MATLAB N4SID，不能冒称原样复现，也不能据此评价 N4SID 的一般表现。现有证据仍不支持稳定算法胜出或已验证的新算法主张。

复算前先按 README 安装主依赖，再运行上面的三条命令。独立核验器会重拟合 8 个源端候选，并重放 3,066 个目标预测。MIT 只覆盖项目编写的代码和说明文字；水箱数据、结果表及上游 NFourSID 源码保留各自的许可边界。

# Cross-Condition Control System Identification Benchmark v1.0.0

**Tag:** [`control-benchmark-v1.0.0`](https://github.com/cyyc0310/tran/releases/tag/control-benchmark-v1.0.0)
**Repository:** [`cyyc0310/tran`](https://github.com/cyyc0310/tran)
**Package:** `system-identification-benchmark/`
**Release date:** 2026-11-03

## What's included

- InTaSet M1↔M2 transfer protocols, source data, scripts, archived results, source hashes, and clean-copy replay receipt.
- The independent Cascaded Tanks physical-system dataset, frozen sparse-input protocol, result files, source verification, and clean-copy replay receipt.
- Supplementary Nano-Drone evaluation reports and receipts. Its raw data, author code, scalers, and checkpoints are not included; the licenses for those upstream materials were not verified.
- ARX/linear state-space and SINDYc-style baselines, independent controlled Neural ODE/hybrid implementations, target-prefix calibration, and a separate endpoint-interpolation oracle.

## Reproduce the included core

Follow the environment and command instructions in [`README.md`](README.md). The included InTaSet and Cascaded Tanks source archives and their licenses are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Check per-file sizes and SHA-256 values against [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json).

## What the results support

This release is a reproducible benchmark and evidence report. The current results do not establish a stable method winner across directions and physical systems, and do not support a new-algorithm claim. Test records were inspected during development, so they are not blind confirmations. Cascaded Tanks has one estimation record and one public test record. The experiments use recorded future inputs for offline prediction; they do not establish online or closed-loop control performance or hardware safety bounds.

The InTaSet paper reports an author Neural ODE baseline, but the checked public author materials did not contain its source, weights, or complete training settings. The included Neural ODE is an independent implementation and is not presented as a reproduction of that author model. Nano-Drone's author Phys+Residual model is a hybrid reference, not a Neural ODE.

## License boundary

The MIT license in this package applies to project-authored source code and prose documentation only. Dataset licenses and result-artifact reuse are documented separately; MIT does not relicense them.

---

## 中文说明

本版本公开 InTaSet 双向实验和 Cascaded Tanks 独立物理系统的输入、协议、脚本、结果及复现凭据；Nano-Drone 仅作补充评估，其上游材料未随包发布，相关许可范围尚未核实。

现有结果支持“基准与适用边界”的定位，尚不能证明某个方法能在不同方向和物理系统上稳定胜出，也不支持提出已验证的新算法。目标记录在开发期间已被查看，不能称为盲测。InTaSet 论文提到的作者 Neural ODE 未能从公开材料中复现；本包中的 Neural ODE 是独立实现，不冒称作者模型复现。MIT 仅覆盖本项目编写的代码和说明文字，数据与实验结果的许可边界见第三方声明。

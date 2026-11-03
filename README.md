# Control-System Identification Research

English | [简体中文](README_zh.md)

This repository's current research focus is system identification and model transfer in control engineering. The versioned benchmark compares forecasting and identification methods under sparse state observations and recorded control inputs.

## Current benchmark

**Cross-Condition Control System Identification Benchmark v1.1.0** covers InTaSet M1↔M2 transfer and an independent Cascaded Tanks physical system. It includes protocols, source receipts, scripts, archived results, and an independently reproducible core package.

- [Read the benchmark overview and reproduction instructions](system-identification-benchmark/README.md)
- [Download the v1.1.0 release package](https://github.com/cyyc0310/tran/releases/tag/control-benchmark-v1.1.0)
- [Read the release notes and evidence limits](system-identification-benchmark/RELEASE_NOTES_v1.1.0.md)
- [Inspect the file-level release manifest](system-identification-benchmark/RELEASE_MANIFEST.json)

The compared methods include ARX/linear state-space, SINDYc-style models, an independently implemented controlled Neural ODE/hybrid model, a two-state Python N4SID baseline, and limited target-prefix calibration. Endpoint interpolation is reported separately as an offline oracle. The N4SID result is an added development comparison on one previously inspected record; it is not an exact MATLAB reproduction or a general conclusion about N4SID.

## Evidence status

The current evidence supports a reproducible benchmark and a study of method limits. It does not establish a stable winner across transfer directions and physical systems or validate a new algorithm. Test records were inspected during development, so results are not blind confirmation. The InTaSet author Neural ODE could not be reproduced from the checked public materials; the included Neural ODE is an independent implementation. Nano-Drone is supplementary and its upstream license scope is unresolved.

## License boundary

The MIT license applies to project-authored code and prose in `system-identification-benchmark/` only. Third-party datasets and generated experiment results have separate reuse terms; see the package's [license scope](system-identification-benchmark/LICENSE_SCOPE.md) and [third-party notices](system-identification-benchmark/THIRD_PARTY_NOTICES.md).

## Repository history

Earlier repository materials are preserved in the [legacy project README](README_transcif_legacy.md) and [简体中文版](README_transcif_legacy_zh.md). They are separate from the current control-system benchmark.

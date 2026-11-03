# Cross-Condition Control System Identification Benchmark v1

**Versioned release:** [`control-benchmark-v1.1.0`](https://github.com/cyyc0310/tran/releases/tag/control-benchmark-v1.1.0), published 2026-11-03. This release preserves v1.0.0 and adds a reproducible Python N4SID development comparison. The package lives in its own subdirectory and leaves the repository's other project files intact. See [`RELEASE_NOTES_v1.1.0.md`](RELEASE_NOTES_v1.1.0.md) and [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json).

This benchmark compares system-identification and prediction methods when observations are sparse and the applied inputs are recorded. It is a benchmark package and evidence report, not a claim that a new algorithm has already won across systems.

## Scope

- **InTaSet M1↔M2:** bidirectional transfer between the two four-tank operating modes. State-observation intervals are 0.5 or 1 second, with target calibration budgets of 0, 1, 3, or 5 seconds.
- **Cascaded Tanks external system:** an independent two-tank physical benchmark with a 4-second native sample interval, included source archive, source hash receipt, frozen sparse-input protocol, and separate results. Its target record was inspected during source admission, so it is development evidence, not blind confirmation.
- **Nano-Drone external evaluation:** a separate physical platform used to check whether the protocol's conclusions carry over. Each 0.5-second prediction window starts from a newly observed state and uses the recorded future motor input. This is offline conditional replay, not online control or a long free-running prediction.
- **Methods:** ARX/linear state-space, SINDYc-style models, an independently implemented controlled Neural ODE/hybrid model, a two-state Python N4SID baseline, and target-prefix calibration. Endpoint interpolation is reported separately as an offline oracle.

Scores are reported separately by system and transfer direction. Target test records were inspected during development, so they are not blind confirmations. Rankings vary by system, direction, and calibration budget; the current evidence does not establish a stable winner or a validated new algorithm. InTaSet's paper describes a Neural ODE, but its author implementation and weights were not found in the checked public materials; the implementation here is independent. The Nano-Drone authors' Phys+Residual model is not a Neural ODE.

The added Python N4SID baseline uses the same Cascaded Tanks estimation and test records as the other Cascaded Tanks comparisons. On this one previously inspected test record, its NRMSE is 0.0463/0.7340/16,122.5093 at 4/12/20 seconds; the fitted model has a spectral radius of 1.025. This is a development result, not blind confirmation. The implementation uses the independently maintained Python NFourSID package and a sparse-observation replay contract; it is not an exact reproduction of MATLAB `n4sid` or the paper authors' score. The result does not establish that N4SID generally performs poorly.

## Reproduce

The independently reproducible core uses InTaSet and Cascaded Tanks. Both raw source archives, source receipts, protocols, scripts, and archived results are included; their source licenses are recorded separately. The Nano-Drone result is an additional external evaluation. Its raw data, author code, scaler, and checkpoints are not included, and the license scope for those specific upstream materials has not been verified. Its fetch path is optional and is not needed to reproduce the licensed core results. The benchmark scripts do not grant rights to upstream data, code, scalers, or checkpoints.

The reviewed environment used CPython 3.12.13. In a new environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-control-benchmark-v1.lock.txt

# Recompute InTaSet from the included source archive.
python scripts/run_intaset_aligned_budget_v1.py \
  --output-dir docs/algorithmic-control-v1/results/intaset_local_01 \
  --seeds 20261102,20261103,20261104

# Verify the included Cascaded Tanks source, then reproduce and verify its results.
python scripts/verify_cascaded_tanks_source_v1.py
python scripts/cascaded_tanks_sparse_input_benchmark_v1.py \
  --output-dir docs/algorithmic-control-v1/results/cascaded_tanks_local_01
python scripts/verify_cascaded_tanks_sparse_input_benchmark_v1.py \
  --output-dir docs/algorithmic-control-v1/results/cascaded_tanks_local_01
```

For the supplemental Python N4SID comparison, install its additional plotting dependencies and run the benchmark plus its independent verifier:

```bash
python3.12 -m pip install -r requirements-cascaded-tanks-n4sid-v1.lock.txt
python3.12 scripts/cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
python3.12 scripts/verify_cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
```

The verifier refits the eight source-side candidates and independently replays the saved target predictions. The protocol, result limits, upstream revision, and output hashes are documented in [`CASCADED_TANKS_PYTHON_N4SID_SPARSE_RESULTS_20261103.md`](docs/algorithmic-control-v1/CASCADED_TANKS_PYTHON_N4SID_SPARSE_RESULTS_20261103.md).

The package also preserves the earlier full InTaSet + Nano-Drone replay receipt. Nano-Drone reproduction is optional and depends on materials from its upstream repository whose specific license scope was not verified; see `THIRD_PARTY_NOTICES.md`. The Cascaded Tanks receipt records a clean replay of the external source archive and its score files. Reproduction receipts distinguish archived replay evidence from any new run.

## Data and license boundaries

- Project-authored source code and prose documentation in this directory are licensed under the MIT License. It does not cover third-party datasets, software, models, checkpoints, or the generated result tables; see `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md`.
- The included InTaSet archive is attributed to its authors and identified as CC BY 4.0 in `docs/algorithmic-control-v1/source_manifest.json` and `THIRD_PARTY_NOTICES.md`.
- The included Cascaded Tanks source archive is attributed under CC BY-SA 4.0. Its exact archive/member hashes, citation, and retrieval details are recorded alongside the data.
- The vendored NFourSID implementation is third-party software under its own MIT license, preserved with the source at `docs/algorithmic-control-v1/external-source/nfoursid/`. The project MIT grant does not replace that upstream notice.
- Nano-Drone CSV files, author code, scalers, and checkpoints are not included. The referenced author repository has no verified license covering those specific materials. The fetch script can retrieve them from the upstream repository at a pinned commit, but that behavior does not grant a license to use or redistribute them. See `THIRD_PARTY_NOTICES.md` for the exact boundary.

## Evidence and limits

The reports and result tables under `docs/algorithmic-control-v1/` describe the tested protocols, per-condition outcomes, data receipts, and known limitations. Each physical system is scored separately; the result tables are not one combined leaderboard. The included results support comparison on these records; they do not establish blind generalization, safety limits, closed-loop performance, or cross-platform bitwise reproducibility. Empirical data ranges are not physical safety bounds.

## Citation

Please cite the InTaSet source dataset when using its data: Putri, S. A., Villacrés, D., Raza, N., Iwakin, O., & Moazeni, F. (2025). *InTaSet: A Benchmark Dataset for Data-Driven System Identification and Fault Detection in an Interconnected Water System* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.17652851](https://doi.org/10.5281/zenodo.17652851).

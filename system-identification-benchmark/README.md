# Cross-Condition Control System Identification Benchmark v1

**Status:** public review candidate; not yet a versioned release. This package is being reviewed in a subdirectory of [`cyyc0310/tran`](https://github.com/cyyc0310/tran), leaving the repository's existing files intact.

This benchmark compares system-identification and prediction methods when observations are sparse and the applied inputs are recorded. It is a benchmark package and evidence report, not a claim that a new algorithm has already won across systems.

## Scope

- **InTaSet M1↔M2:** bidirectional transfer between the two four-tank operating modes. State-observation intervals are 0.5 or 1 second, with target calibration budgets of 0, 1, 3, or 5 seconds.
- **Nano-Drone external evaluation:** a separate physical platform used to check whether the protocol's conclusions carry over. Each 0.5-second prediction window starts from a newly observed state and uses the recorded future motor input. This is offline conditional replay, not online control or a long free-running prediction.
- **Methods:** ARX/linear state-space, SINDYc-style models, an independently implemented controlled Neural ODE/hybrid model, and target-prefix calibration. Endpoint interpolation is reported separately as an offline oracle.

Scores are reported separately by system and transfer direction. Target test records were inspected during development, so they are not blind confirmations. Rankings vary by system, direction, and calibration budget; the current evidence does not establish a stable winner or a validated new algorithm. InTaSet's paper describes a Neural ODE, but its author implementation and weights were not found in the checked public materials; the implementation here is independent. The Nano-Drone authors' Phys+Residual model is not a Neural ODE.

## Reproduce

The package includes the InTaSet source archive, its source receipt, the pinned Python dependency lock, benchmark scripts, and the archived result files. The main raw-to-results command also retrieves Nano-Drone materials from the pinned upstream source; those materials are not included in this package, and their license scope has not been verified. Review the notice below before using that optional external source. The benchmark scripts do not grant rights to upstream data, code, scalers, or checkpoints.

The reviewed environment used CPython 3.12.13. In a new environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-control-benchmark-v1.lock.txt

# Retrieve and hash-check the inputs used by the complete replay.
# This includes Nano-Drone materials from its upstream repository.
python scripts/fetch_control_benchmark_v1_data.py
python scripts/fetch_control_benchmark_v1_data.py --verify-only

# Recompute the full benchmark into a new output directory.
python scripts/reproduce_control_benchmark_v1.py \
  --output-dir docs/algorithmic-control-v1/results/reproduction_local_01 \
  --seeds 20261102,20261103,20261104 --threads 1

# Verify the saved run and every output hash.
python scripts/reproduce_control_benchmark_v1.py \
  --output-dir docs/algorithmic-control-v1/results/reproduction_local_01 \
  --verify-only
```

The InTaSet-only runner is `scripts/run_intaset_aligned_budget_v1.py`. It uses the included InTaSet archive and does not require the Nano-Drone sources:

```bash
python scripts/run_intaset_aligned_budget_v1.py \
  --output-dir docs/algorithmic-control-v1/results/intaset_local_01 \
  --seeds 20261102,20261103,20261104
```

The clean-copy replay receipt describes the earlier full replay. Preparing this public-review package changed release documentation and licensing metadata only; this candidate has not been scored again after those metadata changes.

## Data and license boundaries

- Project-authored source code and prose documentation in this directory are licensed under the MIT License. It does not cover third-party datasets, software, models, checkpoints, or the generated result tables; see `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md`.
- The included InTaSet archive is attributed to its authors and identified as CC BY 4.0 in `docs/algorithmic-control-v1/source_manifest.json` and `THIRD_PARTY_NOTICES.md`.
- Nano-Drone CSV files, author code, scalers, and checkpoints are not included. The referenced author repository has no verified license covering those specific materials. The fetch script can retrieve them from the upstream repository at a pinned commit, but that behavior does not grant a license to use or redistribute them. See `THIRD_PARTY_NOTICES.md` for the exact boundary.

## Evidence and limits

The reports and result tables under `docs/algorithmic-control-v1/` describe the tested protocols, per-condition outcomes, data receipts, and known limitations. The included results support comparison on these records; they do not establish blind generalization, safety limits, closed-loop performance, or cross-platform bitwise reproducibility. Empirical data ranges are not physical safety bounds.

## Citation

Please cite the InTaSet source dataset when using its data: Putri, S. A., Villacrés, D., Raza, N., Iwakin, O., & Moazeni, F. (2025). *InTaSet: A Benchmark Dataset for Data-Driven System Identification and Fault Detection in an Interconnected Water System* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.17652851](https://doi.org/10.5281/zenodo.17652851).

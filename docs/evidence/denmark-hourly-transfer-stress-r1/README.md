# Denmark hourly transfer stress test evidence

Status: scored after prediction file was written and hashed. This is a development extension on a previously viewed data domain, not blind confirmation.

## Frozen inputs

- Target: Energy-Charts Denmark `public_power`, request 2022-01-01 to 2025-01-01; original response SHA-256 `1edf9d721de59e98468bcee0022097159a35912f7a5e022b1ae366a9dc612c12`.
- Source: seven EIA balancing areas; 2022 training, 2023 source calibration.
- Target features: local calendar and t−2 five-group generation mix derived from target 2022–23 hourly rows. This is lag simulation, not an as-issued monthly report.
- The frozen prediction file has 8,779 valid hours across 366 local days; the source response has five incomplete target hours in 2024, retained as missing rather than imputed.
- Prediction SHA-256 `2bd0a8d2e223b1764f3c21793915b27015c8daf9b9f0046902c8ca4a95841ff8`.
- Protocol, runner and shared model hashes are in `prediction_seal.json`.

## Artifacts

- `predictions_2024.csv`: four frozen methods on the common target support.
- `prediction_seal.json`: protocol, model, source/target input and prediction digests; prior target exposure disclosed.
- `scored_summary.json` / `scored_summary.csv`: hourly summary and source-calibrated metrics.
- `monthly_metrics.csv`: month-by-month error, coverage and interval score.
- Runner: [`scripts/run_denmark_transfer_stress.py`](../../../scripts/run_denmark_transfer_stress.py).
- Protocol: [`DENMARK_HOURLY_TRANSFER_STRESS_PROTOCOL_20261102.md`](../../DENMARK_HOURLY_TRANSFER_STRESS_PROTOCOL_20261102.md).

## Label caveat

The response has no hydro or nuclear series. Nuclear is a structural zero in Denmark; hydro is not reported and is provisionally set to zero in this proxy. Danish official sources put 2024 hydro near 0.1% of generation, so the missing zero-emission generation changes the annual denominator only slightly, but its hourly allocation is unavailable. This proxy must not be compared with official consumption intensity or treated as a measured emission label.

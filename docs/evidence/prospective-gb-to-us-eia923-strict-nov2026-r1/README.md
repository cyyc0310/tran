# GB→US November 2026 strict EIA-923 prospective forecast

Status: **sealed, not scored**. Frozen protocol and limitations: [`docs/PROSPECTIVE_GB_TO_US_EIA923_STRICT_NOV2026_RESULTS.md`](../../PROSPECTIVE_GB_TO_US_EIA923_STRICT_NOV2026_RESULTS.md).

- `GB_to_US_EIA923_Nov2026_predictions.csv`: 1,442 point forecasts, 721 continuous UTC hours for each of ERCO and FPL, including the U.S. DST fallback hour.
- `prediction_seal.json`: SHA-256 for the sealed CSV and method/target summary.
- `target_input_audit.json`: exact official EIA-923 August release input rows, completeness, five-group shares, and the explicit reported-plant-set coverage limitation.
- `source_audit.json`, `source_calibration.json`, `startup_hashes.json`, `status.json`: source samples, interval calibration, and code/data provenance.

The target EIA-930 response was not read before sealing. The delayed scorer requires 2026-12-03 UTC or later and exactly 721 complete hours per BA. The eventual response is a current-revised fixed-EF generation proxy. This is a known-geography, one-month prospective check, separate from the seven-BA consumption-side forecast and the US→GB NESO forecast.

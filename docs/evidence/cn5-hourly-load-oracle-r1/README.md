# CN5 hourly load oracle evidence

Development-only information ceiling experiment, 2026-11-02. Protocol: [`CN5_HOURLY_LOAD_ORACLE_INFORMATION_PROTOCOL_20261102.md`](../../CN5_HOURLY_LOAD_ORACLE_INFORMATION_PROTOCOL_20261102.md). Interpretation: [`CN5_HOURLY_LOAD_ORACLE_INFORMATION_RESULTS_20261102.md`](../../CN5_HOURLY_LOAD_ORACLE_INFORMATION_RESULTS_20261102.md).

## Files

- `prediction_seal.json`: nine-country prediction paths, row counts and SHA-256.
- `*_predictions.csv.gz`: sealed `ratio_recomposition`, same-support CN5 `factor_history`, and `load_oracle_factor_history` point forecasts.
- `monthly_metrics.csv`: paired target-month MAE and daily-shape MAE.
- `country_metrics.csv`: within-country equal-month metrics by method.
- `macro_metrics.csv`: country-equal summary.
- `verification.json`: prediction hashes, same-support row/order checks, model reproduction deltas, input hashes and protocol/code hashes.

## Main finding

Same-support country-equal monthly MAE was 62.784 for the CN5 factor head and 62.943 for the same head with actual target-hour load. The Oracle did not meet the preregistered 5% improvement gate. The load series is unavailable at the target-hour forecast origin, so this is only an upper-bound diagnosis. No interval scores were computed.


# Matched BA-month EIA-923 information-value evidence

This package contains the machine-readable result of the 20 BA-month diagnostic described in [`docs/EIA923_ACTUAL_BA_INFORMATION_VALUE_RESULTS_20261101.md`](../../EIA923_ACTUAL_BA_INFORMATION_VALUE_RESULTS_20261101.md).

- `ba_month_method_metrics.csv`: method metrics for each BA-month and input arm.
- `macro_metrics.csv`: equal-weight BA-month summary.
- `pooled_metrics.csv`: hourly pooled summary, with daily shape centered within each BA and local day.
- `hourly_predictions_and_labels.csv.gz`: every scored fixed-EF proxy hour and both arm predictions.
- `provenance.json`: current input/code hashes, runtime versions, recomputed 20-cell eligibility, and old-audit drift counts.

Re-run with `python scripts/run_eia923_actual_ba_information_value.py`. The runner refuses to overwrite its local output directory. The result is an after-the-fact alignment/information-value diagnostic: the August EIA-923 issue postdates evaluated months, and the 90% intervals do not guarantee target coverage.

# Fine-fuel static versus hourly-profile decomposition

Machine outputs for the post-hoc attribution diagnostic described in [`EIA923_FINE_FUEL_PROFILE_DECOMPOSITION_PROTOCOL_20261101.md`](../../EIA923_FINE_FUEL_PROFILE_DECOMPOSITION_PROTOCOL_20261101.md).

- `ba_month_method_metrics.csv`: paired method metrics by BA-month for M5, actual EIA-923 M+F source profiles, and the static EIA-923 fine-share × fixed-EF anchor.
- `macro_metrics.csv`: equal-weight BA-month summary.
- `provenance.json`: links to the earlier frozen 20-cell prediction output and records code/input hashes and the post-hoc status.

Re-run with `python scripts/analyze_eia923_fine_fuel_profile_decomposition.py`. This reuses previously inspected target cells; it is explanatory development evidence, not independent confirmation. EIA-923 postdates the target months and exceeds China's five-group monthly information budget.

# Fine fuel composition increment evidence

Machine outputs for [`EIA923_FINE_FUEL_COMPOSITION_INCREMENT_RESULTS_20261101.md`](../../EIA923_FINE_FUEL_COMPOSITION_INCREMENT_RESULTS_20261101.md).

- `ba_month_method_metrics.csv`: paired metrics by BA-month and information arm.
- `macro_metrics.csv` and `pooled_metrics.csv`: equal BA-month macro and hourly-pooled summaries.
- `hourly_predictions_and_labels.csv.gz`: predictions and fixed-EF proxy labels for all 14,592 paired hours.
- `source_calibration.json`: source-only calibration and per-fuel 2022 source support; zero-supported fuels use a flagged flat fallback.
- `provenance.json`: data/code hashes, software versions, sample definition, and scope limitations.

Reproduce from the repository root with `python scripts/run_eia923_fine_fuel_composition_increment.py`. The runner refuses to overwrite an existing local result directory. The EIA-923 August 2026 issue postdates the evaluated months; M+F exceeds the China five-group information budget, and the EIA-930 oracle uses same-month target fuel mix.

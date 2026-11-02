# Japan CN5 external transfer evidence

Formal run: `results/japan-cn5-external-transfer-r3/`. Forecasts are generated from US/GB source-only training (2022), source-only calibration (2023), and Japan target t−2 five-group shares. Japan's hourly common fixed-factor response is recomputed only after the prediction file and calibration are sealed.

Dataset source: Ihara et al., [Zenodo record 18396311](https://doi.org/10.5281/zenodo.18396311), v1, CC BY 4.0. [Acquisition manifest](acquisition_manifest.json) stores record metadata, published MD5 values, and local README/auxiliary archive SHA-256. The feature archive SHA-256 is `5ef957245f412e8c00aa1fc849bc8c42bd791076603a23129cfcfb95d7cdb712`.

## Evidence files

- `sealed_predictions.csv.gz`: 65,664 local-hour predictions, 9 Japanese TSO areas, 2024-06 through 2025-03.
- `source_calibration.json`: source-only bias and dispersion for each frozen method.
- `prediction_seal.json`: prediction/calibration and input/code/protocol hashes, feature budget, and method list.
- `target_common_fixed_ef_labels.csv.gz`: the post-seal target response reconstructed from hourly generation mix with the source's fixed direct factors.
- `region_metrics.csv`, `monthly_metrics.csv`: per-area and area-month results.
- `area_macro_metrics.csv`, `pooled_metrics.csv`: equal-area macro and hour-pooled summary. Since every area has identical hourly support, point/probability metrics match; shape error is centered within area-day.
- `audit.json`: target support, category mapping, negative-value handling, source data support, label formula, and status.

Reproduce from repository root:

```bash
python scripts/run_japan_cn5_external_transfer.py \
  --output results/japan-cn5-external-transfer-replay
```

The target is a common-factor reconstructed generation proxy, not an official or measured Japan CIF. The public FY2024 hourly generation archive also supplies the ingredients for constructing the target proxy; source models and calibration do not use Japanese labels. The result is a frozen-method external-system stress test, not a fully blinded label release or an as-issued monthly-input replay.

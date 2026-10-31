# V368 protocol: thermal-share-only transfer under the China information budget

## Question

Does the local monthly thermal share alone add transferable value when hourly shapes come from other regions? This directly restricts the local predictor to the coarse thermal/total field available in the NBS provincial table.

## Frozen data and proxy target

- Use the same seven-region 2022–2024 EIA-derived files and valid-hour rule as V367.
- For each held-out region, construct donor hourly CIF proxy profiles from the other six regions in the same calendar year-month. No held-out-region hourly values enter its shape prior.
- Evaluate against the held-out region's fixed-factor hourly eight-bucket CIF proxy. It is derived from generation and is not a measured emissions label.
- The target region contributes only monthly thermal share `s=(coal+gas+oil)/all eight reported categories`.
- Assign one common thermal effective factor `e_T` across the target month and zero direct factor to nonthermal classes. The primary scenario is 700 g/kWh, the midpoint of V364's illustrative `[400,1000]` box; 500 and 900 g/kWh are prespecified sensitivities.

## Paired predictors

For the same donor shape deviation `delta_h = mean_donor_CIF(h) - mean_donor_CIF`, compare:

- `pooled_share_plus_donor_shape`: `clip(e_T * s_donor + delta_h, 0, 1000)` where `s_donor` is the generation-weighted share pooled from the other six regions;
- `local_share_plus_donor_shape`: `clip(e_T * s_local + delta_h, 0, 1000)` where `s_local` is the held-out region's monthly thermal share.

The only difference in the principal pair is local versus leave-one-region-out pooled monthly thermal share. Flat-anchor arms are diagnostics; they do not enter the principal comparison.

## Metrics and analysis

- Thresholds: strict `CIF > τ`, for `τ ∈ {300,500,700}` g/kWh.
- Report hourly proxy CIF MAE and absolute monthly exceedance-count/fraction error, paired by region-month.
- Primary inference: equal weight to each region after averaging its region-month score differences; percentile bootstrap resamples the seven region clusters 10,000 times with seed 368.
- Report each region's mean gain, number of region-month pairs, valid hours, and valid-hour fraction. Missing donor timestamps use zero shape deviation.
- No calibration or tuning on target-region outcomes. No probability-coverage claim.

## Interpretation gate

The local-share claim is supported only if local share improves over pooled share under the identical donor shape and the improvement is not restricted to one region or a single emission-factor scenario. Threshold-specific nulls are substantive: a lower monthly mean can improve level estimation without helping high-threshold tail counts. This remains an EIA proxy-domain test; it cannot validate China CIF accuracy or address above-size versus all-plant generation scope.

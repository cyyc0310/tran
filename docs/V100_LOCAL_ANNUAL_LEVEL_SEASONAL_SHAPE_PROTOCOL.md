# V100: lagged local annual level plus transferable seasonal shape

Frozen 2026-10-31 before running. Hypothesis: when a target region lacks historical hourly CIF, a legally available lagged annual fuel composition may identify the local low-frequency level, while source regions transfer only the monthly deviation around their own annual level. This tests the user's limited-statistics setting on public Chinese province-month proxy data; it does not validate hourly CIF or certify Ember annual detail as an official NBS field.

## Data and target

- Input is the frozen V60 Ember China province-month panel, 31 provinces, 2016–2026. It is a public secondary estimate without a historical release-vintage chain. Its fuel categories are used as a controlled coarsened-data proxy, not as measured hourly emissions.
- The observed monthly coarse information is total generation plus hydro, nuclear, wind, and solar. Define coarse thermal as coal + gas + other fossil + bioenergy.
- The hidden threshold-side target is monthly high-side share `(coal + gas) / thermal` for the sensitivity threshold `tau=300 gCO2/kWh`, under V99 factor intervals where coal and gas are definitely above 300, bioenergy is below, and other fossil is ambiguous. The ratio is a proxy target, not an hourly exceedance count.
- The local added-information budget is the preceding calendar year's energy-weighted `(coal + gas) / thermal` share. This is an explicit sensitivity arm; public official availability of this detailed annual field has not been established.
- Fit source-region models on 2017–2021, calibrate on 2022–2023, and evaluate leave-one-province-out on 2024 and 2025. No target-region monthly fine fuel share enters fitting/calibration. The annual target input comes only from the immediately preceding year.

## Fixed methods

1. `pooled_month`: source-only same-calendar-month average, no target annual fine input.
2. `local_annual_anchor`: target's lagged annual high-side share copied to each month.
3. `annual_plus_source_season`: target lagged annual level plus the source-only calendar-month mean of source monthly share deviations from each source's lagged annual share.
4. `annual_coarse_ridge`: Ridge(alpha=10) on current coarse generation shares, month harmonics, year, and legal lagged local annual share.
5. `annual_coarse_ET`: ExtraTreesRegressor with 300 trees, leaf size 25, max_features .8, same features.

All models use the same source regions and fit/calibration windows. For each arm report point MAE/RMSE in share units and MWh/TWh error after multiplying by observed monthly thermal energy. Report both marginal 90% absolute-residual calibration and a stricter 90th percentile of source-region maximum calibration error. Neither is a finite-sample guarantee for arbitrary distribution shift.

## Decision gate

The level-shape candidate is only a useful advance if `annual_plus_source_season` beats both the local annual-copy baseline and no-local-annual baseline in paired leave-province-out error, without a material coverage collapse. A strong macro average with a failing worst province is not deployment success. Required reports include per-province results, equal-province macro metrics, generation-weighted error, paired province-cluster bootstrap intervals, and worst-region interval coverage. This panel cannot establish hourly coverage, China official-source availability, or method novelty by itself.

## Reproducibility

`python scripts/v100_local_annual_level_seasonal_shape.py`

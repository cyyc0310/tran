# V300 protocol: Can lagged monthly generation mix recover the official province-year CO₂ level?

Frozen 2026-10-30 before analysis. This is a new China-facing experiment motivated by the official 2023 provincial electricity CO₂ factors published by MEE/NBS on 2025-12-31. It tests an annual level-anchor route for cold-start provinces. It does not test hourly CIF or exceedance coverage.

## Target and source contract

- Targets: official 2021, 2022, and 2023 province-average electricity CO₂ factors, kgCO₂/kWh, for 30 provincial grids in the official annex. The 2021–2023 tables are transcribed to `data/china/mee_provincial_grid_co2_factors_2021_2023.csv` from the three MEE/NBS announcements linked below.
- Inputs: Ember's monthly provincial generation by source, rolled to the previous calendar year's annual generation shares for each target year. Exclude aggregate rows and use only the eight atomic categories in the frozen China monthly audit.
- Ember monthly values are retrospective estimates without an archived as-of vintage in this project. The one-year lag is enforced by period, but this experiment cannot certify actual historical release-time availability.
- The official MEE/NBS factor is a province-average electricity CO₂ factor for consumption accounting. It is not an hourly generation-side label, and its system boundary includes factors/components beyond local generation mix. Results must be reported as annual-anchor transfer only.

## Frozen cold-start experiment

Primary test: predict each 2023 official province factor with leave-one-province-out evaluation. The held-out province's 2021–2023 official factors are excluded from fitting. Train on 2021–2022 official targets and lagged annual source shares (2020 mix for 2021 targets; 2021 mix for 2022 targets); predict 2023 from the held-out province's 2022 mix. No 2023 target-year generation mix, official factor, or Ember emissions/intensity field may enter the prediction.

Compare these prespecified predictors:

1. Equal-province training mean.
2. Nonnegative least squares on eight lagged source shares (coal, gas, other fossil, bioenergy, hydro, nuclear, wind, solar), with no intercept, so the fitted prediction is a nonnegative mix-weighted factor.
3. Same nonnegative model after collapsing sources to the five public-style groups thermal, hydro, nuclear, wind, solar.
4. Retrospective local-lag oracle diagnostic: the same province's official 2022 factor. This is not a legal 2023 as-of baseline because the 2022 factor was published on 2024-12-26; report it only as an upper-information reference.

Report province-equal MAE, RMSE, signed bias, 90th-percentile absolute error, paired error differences versus the global-mean baseline, and all 30 province-level residuals. Do not tune on the 2023 target. As a separate descriptive diagnostic, report official province factor changes from 2021→2022 and 2022→2023. No forecast of hourly shapes, no synthetic labels, and no inferential claim from Ember's own emissions/intensity columns.

## Interpretation gates

- A useful level-transfer signal requires the full eight-source model to reduce province-equal MAE by at least 10% versus the training mean and not worsen RMSE; show residuals by province and do not hide worst-case errors.
- A generation-mix-only mechanism is weak if the five-group or eight-source model does not improve on the mean, or if any apparent gain depends on target-year shares.
- Even a pass supports only an annual province-level CO₂ level anchor. Hourly CIF still needs an independently tested temporal-shape/risk model and hourly target labels.
- The 2023 label is published after the target year (2025-12-31). Therefore it cannot be described as an available contemporaneous forecast input; the main result is a spatial cold-start transfer under lagged mix information.

## Official primary sources

- MEE/NBS 2021 factor announcement (published 2024-04-12): https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202404/t20240412_1070565.html
- MEE/NBS 2022 factor announcement (published 2024-12-26): https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202412/t20241226_1099413.html
- MEE/NBS 2023 factor announcement (published 2025-12-31): https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/t20251231_1139517.html
- 2023 factor-use and boundary explanation: https://www.mee.gov.cn/ywdt/zbft/202601/t20260105_1139911.shtml

The MEE/NBS documents say these factors are for electricity-consumption CO₂ accounting and the 2023 methodology/source is consistent with 2021–2022. This experiment does not reinterpret them as operational hourly generation factors.

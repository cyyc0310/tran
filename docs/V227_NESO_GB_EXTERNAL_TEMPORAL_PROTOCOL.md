# V227 Frozen protocol: external temporal check of partial-information CIF risk bounds

Date: 2026-10-30. Status: frozen before computing target-window method scores.

## Question

Does V225's monthly-margin-conditioned feasible-score method retain its field-compression and interval-risk behavior on a different operator dataset, outside the EIA hourly balance family?

This is an external **temporal** check on the Great Britain national generation mix. It is not a cold-start region test and cannot establish Chinese operational validity.

## Data and estimand

Use the National Energy System Operator's public historic GB generation-mix CSV, whose source page reports data from 2009 and an NESO Open Data Licence. Use only rows whose calendar date has 48 half-hour records, aggregate the two half-hour average-power readings to hourly average power, and retain complete naive-calendar days as present in the source. Do not infer a timezone from the naive labels; exclude dates with anything other than 48 half-hour rows, so daylight-saving transition dates do not enter.

Construct domestic generation categories from `COAL`, `GAS`, `BIOMASS`, `OTHER`, `NUCLEAR`, `HYDRO`, `WIND + WIND_EMB`, and `SOLAR`. Exclude `IMPORTS` and `STORAGE` because this test targets domestic generation-side CIF. Categories and proxy EFs are fixed before scoring: coal=1000, gas=400, oil=800 (zero observed oil margin), other/biomass=400, nuclear/hydro/wind/solar=0 gCO2/kWh. This is a transparent fixed-factor proxy, not NESO's reported consumption-side carbon intensity and not metered emissions.

For each target month, derive fuel margins and the energy-weighted monthly anchor from the same complete-day population used for target scoring. Therefore the task is post-month reconstruction, not a real-time forecast.

## Frozen target and source windows

- Targets: all eligible months in 2023 and January–March 2024.
- Source risk pool: same calendar month in earlier years only (2009 through the year before target), excluding target observations.
- Eligibility: target and each source month must contain at least 20 complete days; a target month must contain at least 24 complete naive-calendar days and 12 calibration days.
- Sampling: 12 target days sampled uniformly without replacement, 100 fixed-seed repetitions per target month.
- Calibration unit: one entire day. Nonconformity is the maximum over 24 hourly proxy CIF values or feasible-set endpoint deviations from the monthly anchor.
- Miscoverage: alpha=0.10. Use finite-sample rank `ceil((n+1)*(1-alpha))`, with the maximum observed score when the rank exceeds n.

## Fixed comparisons

1. `source_only`: historical same-month daily-max error scores from earlier years; no target-day hourly channels.
2. `full8_local`: exact target sampled-day hourly fuel mix, used only for local risk calibration.
3. `four_field_local`: target sampled-day hourly domestic total + thermal aggregate + coal + gas. The monthly margins and these fields define a sharp hourly feasible CIF interval; the day score is the maximum endpoint deviation from the monthly anchor.

All methods use the same monthly anchor and evaluate on the unsampled days. Report hourly coverage, simultaneous 24-hour day coverage, mean interval width, and Winkler-90. Compare four-field against full-eight and source-only with paired bootstrap over target months. Do not tune fields, sample budget, EF, target period, or thresholds after looking at results.

## Interpretation gates

Call the external check supportive only if:

- at least 12 target months are eligible;
- four-field and full-eight both have at least 90% macro simultaneous-day coverage;
- four-field has lower macro Winkler than source-only;
- four-field width and Winkler are within 5% of full-eight;
- report month-level uncertainty and all caveats below.

If any gate fails, report it as a failed replication without post-hoc changes.

## Limits

- This tests one national system, not many unseen regions.
- Label CIF is computed from reported fuel mix under fixed proxy EFs; it shares inputs with the information bounds and is not an independently metered emissions label.
- The NESO CSV is cleaned/revised and the currently downloadable snapshot does not provide historical publication vintages. The experiment is not as-of forecasting.
- Target-month margins are post-month, and local calibration requires 12×24×4 hourly aggregate readings.
- Imports/storage and consumption-side network flows are outside the estimand.
- A pass would only show that the mechanism is not specific to EIA's fuel-generation file format. It would not validate Chinese data access or CIF accuracy.

## Frozen input

Official NESO portal: https://www.neso.energy/data-portal/historic-generation-mix

Direct CSV: https://api.neso.energy/dataset/88313ae5-94e4-4ddc-a790-593554d8c6b9/resource/f93d1835-75bc-43e5-84ad-12472b180a98/download/df_fuel_ckan.csv

# V334 protocol: provincial heat-rate information for cold-start annual factor transfer

Registered before running on 2026-10-30. V245–V247 showed that plant conversion efficiency materially affects fuel-specific emissions transfer. V329–V333 showed that generation mix alone carries an annual transfer signal but has region failures, and finer fuel categories alone do not solve them. V334 tests a China-relevant local efficiency statistic in addition to monthly generation mix.

## Data and feature contract

Use MEE 2021–2023 provincial consumption-factor targets, the V320 NBS monthly generation panel, Ember monthly mix, and the 2024 *China Electric Power Statistical Yearbook* provincial 6000-kW-and-above tables. The yearbook reports (a) generation standard-coal consumption rate, table 2-41, and (b) supply standard-coal consumption rate, table 2-43, in gce/kWh for 2019–2023. Use only lagged 2020–2022 values for target years 2021–2023. Tibet is outside the 30-province MEE target cohort.

For each source mix, compare the original five-group vector with two heat-rate-adjusted vectors. Replace thermal share `s_T` by `s_T × r/300`, where `r` is the lagged provincial rate in gce/kWh; leave hydro, nuclear, wind, and solar shares unchanged. The 300 divisor is a fixed scale constant. Fit nonnegative least squares without intercept, with the same years and provinces in every method. This is an efficiency proxy, not a direct carbon factor; 6000-kW-and-above scope and monthly above-designated-size NBS scope are not identical.

## Evaluation

Run both province leave-one-out and V329's frozen seven whole-region holdouts. For 2022 targets, train on 2021 labels and lagged 2020 features; for 2023, train on 2021–2022 labels and lagged 2020–2021 features. No held-out province or region target label may enter fitting. Report MAE/RMSE/P90 by year, pooled province-weighted and equal-region, per-region folds, and paired cluster-bootstrap differences for each heat-rate adjustment versus its same-source five-group baseline.

## Decision and scope

The local-efficiency field advances only if it improves on mix-only under both province and whole-region evaluation, across both target years, with gains not concentrated in one region. Report the two coal-consumption definitions separately. These are retrospective estimates of annual MEE consumption factors; the yearbook vintage was published after the target years, so this is not a legal as-of operational forecast. It does not validate hourly CIF, interval coverage, or the user's table.

## Source and transcription

Source: China Electricity Council, *China Electric Power Statistical Yearbook 2024*, tables 2-41 and 2-43, provincial 6000-kW-and-above generation/supply coal-consumption rates, 2019–2023; source PDF: https://pdf.dfcfw.com/pdf/H3_AP202512311811775347_1.pdf . The V334 input CSV records the 2020–2022 values used. Readable full table extracts are indexed in the source record; the dataset is a manual transcription and must be checked against the page image before any publication claim.

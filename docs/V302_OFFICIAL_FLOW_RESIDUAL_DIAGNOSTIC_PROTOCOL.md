# V302：省级年度电量交换统计能否解释官方EF迁移残差？

Frozen 2026-10-30 before analysis. This is a post-hoc mechanism diagnostic of the already completed V301 2022 leave-one-province-out predictions. It is not a new blind test and cannot establish causal import effects.

## Question and fixed evidence

V301 predicts each 2022 MEE/NBS province consumption factor from its 2021 Ember fuel mix and other provinces' 2021 targets. Use its frozen province-level predictions and residuals without refitting that baseline. Join the 30 factor provinces to the 2022 province annual power table transcribed from *China Electric Power Statistical Yearbook 2024*: total social electricity consumption, gross generation, and cross-province outgoing electricity. The table's generation series is attributed to the China Energy Statistical Yearbook in the yearbook note.

Compute only prespecified descriptors:

1. `balance_gap_pct = 100 * (consumption - generation) / consumption`. This is a supply-demand balance proxy, **not** observed net imports: statistical scope, plant use, grid losses, and exchange flows prevent that interpretation.
2. `sendout_share_pct = 100 * (cross-province sendout / consumption)`, after unit conversion. Outgoing flow alone does not measure imports or the carbon content of traded electricity.
3. Spearman association of each proxy with signed and absolute V301 coarse-five residual.
4. A diagnostic leave-one-province-out linear residual correction using each proxy separately. For each held province, fit an intercept and one standardized feature to the other 29 V301 residuals; add the held-province correction to the frozen V301 prediction. This remains retrospective because both label-year power descriptors and EF labels are from 2022. It is a small-sample diagnostic, not a candidate forecast.

No feature selection, subgroup selection, extra covariates, or 2023 outcomes. Report all 30 rows and errors. Do not claim causality or prospective availability.

## Source and transcription contract

Source: CEC, *China Electric Power Statistical Yearbook 2024*, tables 2-11 (gross generation, 100 million kWh), 3-1 (social electricity consumption, 100 million kWh), and 3-10 (interprovincial outgoing power, 10,000 kWh). PDF mirror used for extraction: https://pdf.dfcfw.com/pdf/H3_AP202512311811775347_1.pdf . The source's table note says generation data come from *China Energy Statistical Yearbook*. The CSV excludes Tibet because the MEE/NBS factor cohort in V301 has 30 provinces.

## Decision rule

If neither direct descriptor has a stable, directionally interpretable association and the leave-one-province-out correction fails to improve absolute error without materially worsening RMSE, stop treating aggregate annual sendout/balance as the next mechanism. The next information frontier is province-resolved inflow with origin/source carbon intensity (or a defensible local hourly generation-side target), not adding more generic annual volume features.

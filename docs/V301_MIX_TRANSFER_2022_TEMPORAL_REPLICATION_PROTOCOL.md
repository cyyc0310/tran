# V301 protocol: one-year training replication of annual mix-to-factor transfer

Frozen 2026-10-30 after V300's 2023 cold-start result. This is an explicitly exploratory temporal replication on the same official MEE/NBS factor series and Ember input family, not an independent data-source confirmation.

## Contract

- Hold out each of the 30 MEE provincial grids in turn.
- Fit on the other 29 provinces' 2021 official factors and lagged 2020 Ember annual generation mix.
- Predict the held-out province's 2022 official factor using its 2021 Ember annual mix.
- Compare equal-province mean, nonnegative five-group mixture (thermal/hydro/nuclear/wind/solar), nonnegative eight-source mixture, and the delayed local-2021-factor reference. The latter is an information-value reference, not an as-of 2022 predictor.
- No target-2022 generation mix, 2022 EF, or Ember emissions/intensity column may enter fitting.
- Report province-equal MAE/RMSE/bias/p90 absolute error and all province residuals. No tuning based on this replication.

All source-boundary and vintage limitations in V300 apply. V301 cannot verify hourly CIF or forecast interval coverage.

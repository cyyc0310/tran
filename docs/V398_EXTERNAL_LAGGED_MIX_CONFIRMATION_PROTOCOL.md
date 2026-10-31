# V398: frozen lagged-mix model on external EIA regions

Frozen before opening external 2026H1 hourly labels. The external six are AZPS, DUK, LGEE, PACE, PSCO, and SRP from the existing controlled-information dataset; the only target hours scored are 2026-01-01 through 2026-06-30 UTC.

## Frozen model

- Source regions: the seven EIA areas in V369/V395: BPAT, CISO, ERCO, FPL, ISNE, MISO, NYIS.
- Train: 2022 hourly fixed-factor proxy labels from source regions only.
- Calibration: 2023 source-region-month blocks. Each calibration region's predictions are cross-fitted from a Ridge model trained on the other six source regions' 2022 data.
- Test: six external regions' 2026H1 full UTC-hour grid. No external hourly target enters fitting, calibration, or prediction.
- Fixed EF proxy: coal 1000, gas 400, oil 800, and remaining five buckets 0 g/kWh, matching V369's generation-derived target.
- Models: standardized Ridge with alpha 10. Calendar features are cyclic hour, weekday, and month. The local-mix model adds t-2 thermal share and its hour/month interactions. The no-mix model uses the same calendar features without composition.
- Local feature: exact `t-2` share, calculated from the published monthly coal+gas+oil shares in V6 `monthly.csv`, only if that row's `period_end` and `available_at` are no later than target-month start and coverage is at least 80%. If absent, the local-mix arm falls back to the no-mix forecast and is marked as such.
- Hourly intervals: separate 90%/95% radii from the corresponding quantile of 2023 cross-fitted source-region-month maximum absolute residuals.
- Tail-count intervals: per threshold (300/500/700), alpha, and model, radius is the corresponding 2023 quantile of absolute source region-month tail-fraction error; at scoring time this fraction is multiplied by valid target hours and rounded up.
- Metrics: hourly MAE, interval coverage, mean width, interval score; threshold count coverage, width, and interval score. Report overall, per region, and local-feature-available-only subsets.

## Information and score boundary

Prediction stage may read source 2022/2023 labels, target UTC timestamps, the external V6 audit metadata, and V6 monthly features with `available_at <= origin`. It must not read external hourly generation values, `valid` flags, or target CIF labels. Prediction outputs, month-level calibration radii, and the method specification are hashed into a seal before scoring. Scoring verifies the seal before opening external hourly reference values and scores only valid 2026H1 rows. No tuning follows this confirmation; changes require a new protocol and a new untouched target.

## Limits

This is an external-region test of the EIA fixed-factor proxy, not Chinese measured CIF. The six regions are new relative to V395's seven source areas but have been used in other project experiments; call this a frozen-method confirmation, not a globally untouched dataset. V6 monthly inputs use a controlled month-end plus 30-day availability rule; they are not true original release vintages. Region cluster count is six, and split-block conformal coverage is descriptive under temporal and spatial dependence.

## Reproduction

1. `python scripts/v398_predict_external_lagged_mix.py`
2. Verify `results/v398-external-lagged-mix-r1/prediction_seal.json`.
3. `python scripts/v398_score_external_lagged_mix.py`

# V50 monthly activity-scale feature: source-held-out diagnostic

Registered 2026-10-27 before computation. The prior V50 level-correction probes used fuel shares compressed into one fixed-EF intensity feature and ignored monthly total generation. This round tests whether monthly activity scale adds information beyond fuel composition.

## Fixed design

- Use the same frozen V50 seals, 10 external source regions, seven outer functions, 90-day eligible labels, Scale-anchor raw paths, and [0,1000] factor-2 intervals. Verify the same hashes; no target/unseen artifacts or point/risk refits.
- From the three latest complete monthly records legally released by each query date, build (a) the existing mean fixed-EF fuel-share proxy and (b) the mean monthly average output in MW (`total_mwh / expected_hours`), transformed as `log1p(MW)`. `period_end` and `available_at` must both precede the query date. Reject nonfinite/nonpositive total or expected-hours values.
- For each outer and held source, train equal-source-weighted OLS on the other nine sources' daily mean residual response. Compare no shift, equal-source global bias shift, EF-only, activity-only, and EF+activity (intercept plus the indicated standardized feature columns). Standardize feature columns from the training rows only, using the same source-equal weights; constants map to zero standardized values.
- Apply the fitted daily offset uniformly to all 128 hourly particles, then use the same clipping and factor-2 interval metrics. Report the full source-level metrics and clipping. No post-hoc arm selection is confirmation.

## Interpretation boundary

This tests predictive information in a proxy monthly table, not the validity or publication vintage of any Chinese province's monthly generation series. The ten sources are reused development sources; rows/days do not increase the independent region count. This is a descriptive feature ablation and cannot claim a calibrated transfer rule or novelty.

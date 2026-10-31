# V370 protocol: threshold-side gate for selective tail-count intervals

## Question

Can the analytic threshold-side condition identify when to abstain from a potentially misleading narrow monthly tail-count interval, while issuing shorter calibrated intervals on the non-crossing subset?

## Frozen settings

- Use V367–V369's seven-region 2022–2024 EIA proxy panel and leave-one-region-out outer evaluation.
- Strict event: hourly proxy `CIF > τ`, `τ ∈ {300,500,700}` g/kWh.
- Illustrative effective thermal EF support: `[400,1000] g/kWh`; `s=(coal+gas+oil)/total`.
- A month is gate-crossing if `400*s ≤ τ < 1000*s`. Its risk-count interval is structurally unresolved under the proportional-mix witness and is returned as `[0,H]` or marked abstain.
- The base local point predictor is `700*s_local` plus the donor-transferred hourly CIF deviation from V369.

## Cross-fitted calibration and comparisons

For each outer held-out region, build residuals from only the other six regions. Each calibration region-month's donor shape excludes that region. Compare:

1. local-share global asymmetric residual interval (all training region-month residuals);
2. pooled-share global asymmetric residual interval;
3. gate-conditioned selective method: on non-crossing months, calibrate asymmetric count residual quantiles using only non-crossing training months; on crossing months, either emit `[0,H]` or abstain.

For integer bounds, floor the lower endpoint, ceil the upper endpoint, then clip to `[0,H]`. Use nominal 90% and 95% residual quantile levels. Do not tune on held-out-region labels.

## Metrics and decision rule

Report overall count-interval coverage and width when crossing months are assigned `[0,H]`; for abstention, report retained fraction and count-interval coverage/width only among issued predictions, with region-level breakdown. Compare to the two always-predict baselines. Use interval score and worst-region coverage as secondary metrics.

This experiment is descriptive cross-validation, not a formal conformal guarantee: only seven region clusters, serial dependence, retrospective monthly margins, and a proxy target. The candidate is useful only if non-crossing predictions retain adequate held-out coverage with materially smaller intervals and the crossing set gives a clear, honest abstention signal. Otherwise retain the gate solely as an information warning, not a prediction method.

# V369 protocol: local thermal share with calibrated hourly risk intervals

## Question

Can the local thermal-share level plus a donor-transferred hourly shape produce narrower hourly CIF and monthly tail-count intervals than strong residual-calibrated baselines while retaining empirical coverage?

## Data and outer evaluation

- Use the 2022–2024 seven-region EIA hourly proxy panel and validity rule fixed in V367/V368.
- Outer evaluation is leave-one-region-out (7 folds). In a fold, no target-region hourly CIF proxy enters donor-shape construction or calibration residuals.
- The target's only local feature is the month-level thermal share `(coal+gas+oil)/total`.
- Target is the eight-bucket fixed-factor CIF proxy; it is not measured emissions or full-grid truth.
- Point anchor uses a common thermal EF of 700 g/kWh, the midpoint of V364's illustrative `[400,1000]` interval. The local EF-box interval is included as an uncalibrated information baseline.

## Cross-fitted calibration

For each outer held-out region, build calibration residuals from the other six regions. For each calibration region-month, construct the donor hourly shape from the other five available regions, then form local-share and pooled-share hourly point predictions. Do not use the outer held-out region at any stage of these calculations.

Use these interval methods:

1. `local_ef_box`: `[400*s_local + delta_h, 1000*s_local + delta_h]`;
2. `local_symmetric_q`: local-share point profile plus/minus the training-region empirical `1-alpha` quantile of absolute residuals;
3. `local_asymmetric_q`: local-share point profile plus training-region residual quantiles at `alpha/2` and `1-alpha/2`;
4. `pooled_symmetric_q` and `pooled_asymmetric_q`: same residual calibration, but with the leave-one-region-out pooled thermal-share point level.

Nominal interval levels are 90% and 95%. Clip hourly endpoints to `[0,1000]` g/kWh. The empirical residual quantile is computed with the “higher” order statistic.

## Metrics

- Hourly proxy coverage and mean interval width, averaged first within region-month and then equally over regions.
- For each `τ ∈ {300,500,700}`, map hourly intervals to monthly count bounds `[sum(LO_h>τ), sum(HI_h>τ)]`; report count-interval coverage, mean width in hours and as a fraction of observed month length, and Winkler interval score.
- Give per-region results and worst-region coverage. Report pooled-hour coverage only as a secondary descriptive metric.
- No parameter tuning on outer held-out outcomes. No formal conformal guarantee is claimed: temporal dependence, seven region clusters, retrospective features, and a proxy target violate/leave unverified standard exchangeability and target assumptions.

## Gate for a useful increment

The candidate needs to attain nominal coverage approximately and improve interval width or Winkler score against both local and pooled symmetric/asymmetric residual baselines, without severe worst-region failure. The EF-box interval is informative only as a width/coverage stress baseline. Any passing result remains an EIA proxy-domain result and does not validate China's NBS scope or local factor support.

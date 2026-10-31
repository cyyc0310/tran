# V369 results: cross-fitted hourly and tail-count intervals

Protocol: [V369](V369_INTERVAL_TRANSFER_PROTOCOL.md). Runner: `scripts/v369_crossfit_risk_intervals.py`. Results: `results/v369-crossfit-risk-intervals-r1/`.

## Hourly CIF proxy intervals

In seven-region outer leave-one-region-out folds, local asymmetric residual quantiles were much better than pooled-level residual quantiles, but did not maintain nominal coverage uniformly:

| Nominal | Local asymmetric coverage | Local width | Worst-region coverage | Pooled asymmetric coverage | Pooled width |
|---:|---:|---:|---:|---:|---:|
| 90% | 87.4% | 213.4 g/kWh | 68.2% | 74.6% | 479.2 g/kWh |
| 95% | 92.6% | 246.7 g/kWh | 76.3% | 78.2% | 515.7 g/kWh |

Local monthly level information makes the interval substantially more efficient than the pooled level, but its worst-region coverage is poor. The uncalibrated `[400,1000]` effective-EF box with donor shape covered only 63.8% of hourly proxy labels, so the monthly effective-EF box alone does not describe hourly shape risk.

## Monthly tail-count intervals

At τ=300, local asymmetric calibrated count intervals covered 97.6% of region-months at nominal 90% and 99.2% at nominal 95%, but their equal-region mean widths were 563 and 581 hours—about 79% and 81% of a month. They largely retain the broad ambiguity rather than sharply resolving it.

The threshold-side gate sharply separated subsets in this proxy panel. With `[400,1000]` and local monthly thermal share:

- At τ=300, 203/245 region-months crossed the EF support threshold. Local-asymmetric count intervals covered 98.0% of crossing cases with mean width 94.0% of the month at nominal 90%. Among 42 non-crossing cases, coverage was 95.2% and mean width 2.0%.
- At τ=500, 152/245 crossed. The same intervals covered 85.5% of crossing cases with width 12.1%; among 93 non-crossing cases, they covered 100% and had zero mean width.

This is evidence that `τ/s` relative to local factor support predicts where risk counts remain difficult. It does not by itself yield a calibrated forecasting method: the factor support is illustrative, the regions are already seen, and hourly labels are a fixed-factor proxy.

# V133 — Measurement Error for Multi-Day Risk-Side Summaries

## Question

Does bounded error in the threshold-side MWh field erase the information-efficiency benefit of three-day reporting?

## Frozen design

- Reuse the V129 358 region-month EIA cohort, τ=600/900, factor support, and 1/3/7-day within-month UTC bins.
- Stress relative risk-side field error at 0%, 1%, 2%, 5%, and 10%, using the same symmetric interval convention as V130, clipped to the exact thermal total in each bin.
- Cross-check the closed-form error set against V130's general measurement-error solver on a fixed synthetic grid before scoring.
- Keep daily-bin thermal total and clean total exact. This isolates the single risk-side MWh measurement error; it is not a full uncertainty budget.
- For each block, evaluate exact robust K endpoints across the latent risk-side MWh interval and EF support. Sum lower and upper endpoints by month. Compare with the matching zero-error detailed-fuel oracle and the monthly risk-side baseline.
- Verify nominal fixed-factor K containment, zero-error equality with V132, monotone interval expansion with error, and nesting of the risk-side set around the detailed-fuel robust set. Keep all frozen rows; any failure invalidates the run.

## Interpretation boundary

Error levels are sensitivity values, not empirical Chinese publication-error estimates. This is retrospective EIA oracle coarsening, not Chinese validation or a forecast-time assessment.

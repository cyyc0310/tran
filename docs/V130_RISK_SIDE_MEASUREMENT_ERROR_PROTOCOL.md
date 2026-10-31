# V130 — Risk-Side MWh Measurement Error under EF Support Uncertainty

## Question

Does V129's factor-support-aware threshold-side interval remain informative when the single local MWh field itself has symmetric relative error?

## Frozen comparison

- Use V129's fixed 358 region-month cohort, EF intervals, coarse capacity, strict τ=600/900 thresholds, and 1%/5%/10% tail targets.
- Stress side-field relative error at 0%, 1%, 2%, 5%, and 10%, clipped to the known monthly thermal total.
- At τ=600 the measured field is coal+oil MWh; all allowed high-side EFs exceed τ, and the low-side EF can reach τ. At τ=900 it is coal MWh; coal EF can equal τ, so include the exact K=0 endpoint as well as the upper coal-EF endpoint set.
- Compare the error-robust risk-side set with the V129 robust detailed-margin set and the same coarse-capacity thermal-total set on identical rows.
- Assert nominal proxy K containment, zero-error equality with V129, monotone set expansion with error, and nesting within the coarse thermal-total information set. Record failures and availability rather than changing the frozen cohort.

## Interpretation boundary

Relative error is a sensitivity parameter, not an empirically estimated Chinese reporting-error distribution. Thermal total and capacity remain exact. The study is retrospective EIA oracle coarsening, not China forecast validation.

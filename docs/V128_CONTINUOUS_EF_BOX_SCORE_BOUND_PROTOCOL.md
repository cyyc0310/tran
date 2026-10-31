# V128 — Conservative Calibration Score Bound over the Continuous EF Box

## Question

Does the τ=900 support-gap rule retain useful interval efficiency when the calibration score is protected against every independent coal/gas/oil/other EF vector in the ±20% box, including vectors where the monthly mean crosses τ between vertices?

## Frozen procedure

- Reuse the 2024–2025 source calibration roster, the same 25 source regions, and the October 2026 target cohort from V122. October labels are already open; this remains a posthoc stress analysis.
- Compute exact observed K at the all-low and all-high EF vectors. Because each hour's CIF is coordinatewise nondecreasing in EF, these give the minimum and maximum K over the continuous box for each observed hourly schedule.
- Compute exact monthly-mean CIF at all-low and all-high. These determine whether the mean anchor can only be 0, only 1, or can cross τ within the box.
- For a source month, upper-bound its absolute anchor error for every EF in the box by the maximum applicable term: `K_high/T` when anchor 0 is possible, and `1-K_low/T` when anchor 1 is possible. This is conservative because it ignores dependence between the anchor-crossing surface and K at that EF.
- Take each source region's maximum monthly upper-bound score, apply the same α=10% leave-region-out rank, and combine the radius with V127's exact independent-corner physical endpoint hull and frozen q=20% support threshold.
- Compare anchor-only, capacity-only, plain intersection, and the existing gated decision on the same 37 capacity-feasible October regions.

## Interpretation boundary

The score bound is valid over the continuous independent EF box but may be wider than the true maximum. It protects the EF dimension only; it does not create a time-exchangeability guarantee, and the October target labels are reused. Factors remain illustrative fixed proxies rather than empirically estimated Chinese factor intervals.

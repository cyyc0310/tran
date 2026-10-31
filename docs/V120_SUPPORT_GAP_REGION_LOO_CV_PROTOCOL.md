# V120 — Support-Gap Fallback Region-LOO Cross-Validation

## Question

Does a source-calibrated support-gap fallback generalize when the entire target region is held out, rather than relying on a target-quarter threshold sweep?

## Frozen evaluation

- Use the 25 V111 source-calibration regions and their 2024–2025 monthly records. Each region is an outer test fold. Exclude it from both the conformal error calibration and support-gap boundary calculation.
- For each fold and threshold, calibrate the anchor radius from the other 24 regions' region-level maximum monthly worst-case absolute errors using α=0.10 and rank 23.
- Define normalized monthly conflict as `max(0, capacity_upper_box/hours − anchor_base_upper_fraction)`, where `capacity_upper_box` is the maximum sharp upper K endpoint over the same ±10% EF vertices and the capacity vintage is one year lagged. This uses only target-month coarse inputs and does not depend on the held-out fold's labels or conformal radius.
- Compare a no-fallback anchor∩capacity rule to fallbacks when conflict exceeds a source-region maximum-gap quantile. Predeclare support-tail levels q={0.05, 0.10, 0.20, 0.30}; the boundary is the order statistic `ceil((n+1)*(1-q))` over training-region maximum gaps, with the held-out region excluded.
- Evaluate against exact hourly factor-box K endpoint hulls. Preserve all months with ≥80% valid proxy hours; report capacity-infeasible months separately. Compare pooled and region-equal coverage, width, and fallback rate against capacity-only.

## Interpretation boundary

This is development cross-validation on the existing source pool, not final independent confirmation. Candidate support-tail levels are reported as a frontier; do not select one based on V118 target labels. The ±10% EIA proxy and U.S. capacities are not real Chinese CIF validation.

## Numerical audit amendment before final scoring

The first exploratory run (r1) used floating-point threshold comparisons for hourly proxy labels. Before accepting any metrics, rerun as r2 using exact rational comparisons for the strict event definition, consistent with V109. Preserve r1 as a diagnostic artifact; only r2 is eligible for interpretation.

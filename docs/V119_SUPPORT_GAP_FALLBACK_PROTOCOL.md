# V119 — Calibration-Support Gap Fallback Diagnostic

## Question

Can the V118 failures be flagged without target labels by checking whether the calibrated monthly-anchor upper endpoint is far below the exact physical capacity upper endpoint? If the disagreement is outside the source calibration support, fall back to the capacity set for that month.

## Frozen exploratory diagnostic

1. On each of the 25 V111 source-calibration regions, form monthly fuel margins for 2024–2025 and lagged (year−1) EIA capacity group constraints. Compute the exact ±10% EF-box upper K endpoint under the V92 sharp group-capacity set.
2. For each source region, use the V116 leave-one-region-out conformal radius, and derive the source month’s calibrated anchor upper K endpoint with the same integer rounding as V118.
3. Define normalized conflict `g = max(0, capacity_upper/hours − anchor_upper/hours)`. Freeze each threshold's maximum observed source-region `g` as a support boundary. No target Q3 labels enter the boundary.
4. On V118 target rows, apply the frozen intersection when `g` is within the source support boundary; if `g` exceeds it, output the exact capacity interval for that month. Evaluate proxy endpoint coverage and width against the original intersection and capacity baselines.

## Interpretation boundary

This is exploratory because V118's target cohort and labels are already open. The source-support maximum is a heuristic out-of-support alarm, not a coverage theorem. The source pool is only 25 regions and may not be exchangeable with the target cohort. Any apparent coverage restoration must be frozen and tested on a new region-time cohort. EIA factor-box K remains a fixed-factor proxy, not Chinese real CIF.

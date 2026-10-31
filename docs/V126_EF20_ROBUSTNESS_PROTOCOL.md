# V126 — ±20% Fuel-Factor Uncertainty Stress Test

## Question

Does the frozen q=20% support-gap fallback still improve the coverage–width frontier when thermal EF uncertainty widens from ±10% to ±20% and all source calibration scores are recomputed under the wider box?

## Frozen procedure

- Use the V122 October MTD cohort (44 metadata-selected regions; retain all availability slots) and source calibration month roster from V111.
- Replace the ±10% multiplicative box for coal/gas/oil/other with ±20%; keep nonthermal factors fixed.
- Recompute source monthly mean-anchor base intervals, worst-case absolute-error scores, and exact capacity-support gap features from hourly 2024–2025 calibration records. For each target region, exclude it from both source calibration and support-boundary pools.
- Use anchor α=10% region-level split-conformal rank and support-tail q=20% rank, exactly as V121/V122. Use lagged capacities and the V92 sharp set; all endpoint/hours comparisons use exact rational arithmetic.
- Freeze the same threshold policy: τ=600 capacity-only; τ=900 anchor∩capacity unless support gap exceeds the source boundary, then capacity. Compare anchor-only, capacity-only, plain intersection, and gated fallback.

## Interpretation boundary

This is a factor-uncertainty robustness stress test on the already opened October cohort, not an independent confirmation. The ±20% box is a sensitivity scenario, not a sourced Chinese EF confidence region. Results remain fixed-factor U.S. proxy evidence.

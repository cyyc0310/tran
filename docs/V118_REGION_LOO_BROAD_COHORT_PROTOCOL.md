# V118 — Broad-Cohort Region-LOO Robustness Diagnostic

## Purpose

Stress the frozen V116 all-month region-grouped split-conformal radius plus the monthly-average anchor ∩ exact capacity-feasible interval on a broad 2026Q3 region cohort. Determine whether V116's apparent width gain is confined to the four V115 regions or persists across a wider set, and report failure modes rather than tuning to them.

## Cohort and split

- Select all V52 regions with 2026H1 complete-proxy coverage at least 90%, using only the existing quality census. Include all July–September 2026 slots and report missing/ineligible slots; do not screen using K, factor-box endpoints, or interval results.
- Use the frozen V111 source calibration rows. For each target region, exclude that entire region from calibration if present there; retain all other calibration regions. Thus each scored region has a region-disjoint calibration fold. For 25 remaining source regions use rank 24 at α=0.10; if one of the 25 is held out, use n=24 and rank 23.
- For each threshold, score a source region by its maximum monthly worst-case absolute anchor error across the full V111 calibration period. Use separate threshold radii; no season-specific tuning.
- Keep the V115 factor box, the V90/V92 monthly and capacity constraints, 80% target-month coverage threshold, and anchor-capacity intersection unchanged. Compare monthly-only, capacity-only, anchor-only, and intersection intervals, with factor-box proxy K containment, width, and 1/5/10% decision frontier.

## Interpretation boundary

This is a broad robustness diagnostic, not a blind confirmation: V113/V115 exposed part of 2026Q3 and V116 was designed after V115. The region pool is selected on historical data completeness, so exchangeability conditional on selection is unproven. Report outcomes separately for source-calibration regions, prior V104/V115 targets, and other regions. The data are U.S. fixed-factor EIA proxy labels and lagged U.S. capacities; this does not establish a China CIF result or a month-ahead forecast.

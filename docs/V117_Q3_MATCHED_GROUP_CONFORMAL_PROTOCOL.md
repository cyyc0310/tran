# V117 — Q3-Matched Region-Grouped Conformal Diagnostic

## Question

Does aligning the calibration score window to the July–September target season change the V116 interval-efficiency/coverage tradeoff?

## Frozen diagnostic

- Reuse the V116 region-level maximum absolute error score, but calculate each of the 25 calibration-region scores using only July, August, and September 2025, matching the three target months in V115.
- Keep thresholds separate, with `alpha=0.10` and rank `ceil((25+1)*(1-0.10))=24`.
- Recompute the V115 four-region, 2026Q3 outputs with that radius, the unchanged factor box, capacity constraints, eligibility rule, and interval-intersection rule.
- Compare widths, factor-box proxy containment, and decision frontier to V115/V116.

## Interpretation boundary

This is a post-hoc diagnostic on V115 target labels already opened, not a blind confirmation. The same-season window makes calibration and target score windows more comparable, but does not establish exchangeability across regions or years, especially after selecting target regions on data completeness. Split-conformal coverage is marginal over exchangeable regions; it is not conditional coverage for a specific region. The target is an EIA fixed-factor proxy, not real Chinese hourly CIF.

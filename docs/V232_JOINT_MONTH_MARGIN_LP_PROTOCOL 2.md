# V232 Frozen protocol: joint-month sharp CIF bounds from monthly margins

Date: 2026-10-30. Frozen before computing V232 results.

## Question

V231 used a sharp one-hour feasible set but did not jointly allocate every fuel across every hour of the month. This run tests whether the complete monthly fuel margins, together with hourly total generation for the entire month, materially tighten hourly CIF bounds when all hours are constrained simultaneously.

## Cohort and sampled targets

Reuse V226's 26 BAs and 74 BA-months (2026 July–September). For each BA-month, select one complete calendar day using the fixed RNG seed `232`. The full month's hourly total generation and every fuel's monthly margin enter the joint LP; only selected-day hours receive per-hour min/max queries. This reduces computation while preserving a predefined target-hour sample.

## Joint feasible set

Let `x[t,j] ≥ 0` be fuel `j` at hour `t`. Impose exact hourly totals `sum_j x[t,j]=G[t]` for all month hours and exact monthly margins `sum_t x[t,j]=M[j]` for every fuel. For each sampled hour, minimize and maximize `EF·x[t,:]/G[t]` over this same month-wide feasible set. Compare with V231's pointwise bounds, which do not use other hours' exact totals.

## Reporting

Report pointwise versus joint interval widths, paired reduction, LP feasibility, and nominal-proxy witness containment. This remains fixed-EF proxy identifiability analysis, not calibrated prediction or independent validation. Hourly totals are granted even though the user's Chinese input currently provides monthly data only.

## Stop checks and limitations

- Require at least 30 BA-months and 10 BAs.
- All joint LP extrema must be feasible and bracket the nominal EIA witness within `1e-7`.
- Do not call the sampled-day result a complete-month bound summary.
- The joint LP is sharp only under the stated aggregate monthly and hourly-total constraints; it has no capacity, ramp, imports, storage, or network constraints.

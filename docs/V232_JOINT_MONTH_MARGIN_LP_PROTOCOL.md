# V232 Analytical follow-up: joint-month margins do not tighten the one-hour set

Date: 2026-10-30. The repeated LP plan was stopped before producing results after the exact transportation-polytope equivalence below was established.

## Question and derivation

The question was whether complete monthly fuel margins and every hourly total tighten a selected hour's CIF range beyond V231's one-hour bounds.

Let a target hour have total `G_t`, monthly fuel margins be `M_j`, and target-hour allocation be `x_j`. V231's feasible set is `sum_j x_j=G_t`, `0≤x_j≤M_j`.

For any target-hour vector in that set, the remaining fuel supplies are `M_j-x_j≥0`, and their total equals `sum_{s≠t}G_s`. A nonnegative transportation matrix with these remaining fuel supplies as row sums and remaining hourly totals as column sums always exists when totals agree. Thus every V231 target-hour allocation extends to a full-month allocation. Conversely, every full-month allocation satisfies V231's target-hour constraints. The sets have identical one-hour projections, so their CIF extrema are the same under these constraints.

The repeated LP was canceled as redundant; no V232 solver results are claimed. This proof is conditional on these aggregation constraints and fixed EF. Hourly totals are granted even though the user's Chinese input currently provides monthly data only.

## Conditions and limits

- The identity requires no added per-fuel hourly capacities or temporal/network constraints.
- Fuel-specific installed capacity, ramping, minimum generation, storage, dispatch, imports, and renewable availability constraints can shrink the month-wide set; they are not represented here.
- This remains fixed-factor proxy identifiability reasoning, not calibrated prediction or independent validation.

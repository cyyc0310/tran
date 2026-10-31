# V138 — Measurement Error for Three-Day Risk-Side + Capacity

## Question

Does bounded error in the three-day threshold-side MWh field remove the interval-width and selective-decision gains found in V136 when lagged coarse capacity remains exact?

## Frozen design

- Reuse V136's 358 region-month cohort, three-day bins, lagged coarse capacity and no-capacity fallback, V129 EF support, τ=600/900, and ≥1/5/10% decisions.
- Stress per-block risk-side MWh relative error at 0%, 1%, 2%, 5%, and 10%, using V130's symmetric interval convention, clipped to the known block thermal total.
- Treat thermal total, clean total, and coarse capacity as exact. This isolates risk-side measurement error; it is not a full Chinese data-quality model.
- Compare against V136's zero-error three-day endpoints, monthly detailed-fuel+capacity, and the matching three-day detailed-fuel oracle under the same block capacity/fallback rule.
- Assert zero-error equality with V136, nominal-K and detailed-envelope containment, monotone expansion with error, and complete denominator preservation. Report all failures.

## Interpretation boundary

Error values are sensitivity assumptions, not measured Chinese publication errors. This remains retrospective EIA oracle coarsening; it does not validate actual CIF predictions or local Chinese capacity records.

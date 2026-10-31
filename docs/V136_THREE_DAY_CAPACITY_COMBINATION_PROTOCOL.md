# V136 — Three-Day Risk-Side Summaries with Coarse Technology Capacity

## Question

Does adding lagged coarse technology capacity to the frozen three-day risk-side summary improve the selective risk frontier, compared with monthly detailed-fuel margins plus the same capacity?

## Frozen design

- Reuse V129's 358 region-month cohort, three-day UTC bins, EF support, τ=600/900, and ≥1/5/10% decisions.
- In every month, use the same lagged coarse thermal/nuclear/hydro/wind/solar capacity as V129. Compare the monthly detailed-fuel+capacity bound, monthly risk-side+capacity bound, three-day risk-side+capacity bound, and a three-day detailed-fuel oracle under the same per-block capacity contract.
- A three-day block is capacity-feasible only if every group energy total is at most block hours × lagged group MW. If any block fails, preserve that block and use its no-capacity bound; record the failure count rather than dropping the region-month. Apply the identical capacity/fallback rule to the three-day detailed oracle.
- Sum blockwise integer endpoints and assert nominal K containment, detail-oracle containment, zero failures for any assigned capacity set, and identical denominators. Report the block-level capacity fallback fraction.
- Capacity is EIA-860 proxy information; do not present it as validated Chinese provincial data availability.

## Interpretation boundary

This is an oracle-coarsening information-value experiment, not a forecast and not a Chinese validation. The method is only deployable where capacity vintage, technology mapping, statistical boundary, and as-of availability are independently verified.

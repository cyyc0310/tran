# V129 — Factor-Support-Aware Risk-Side Monthly Information

## Question

Does a threshold-indexed monthly energy summary remain close to detailed fuel margins under wider, fuel-specific EF intervals, when compared on the same EIA region-months and the same coarse technology capacities?

## Frozen cohort and factor intervals

- Use the fixed 358 region-month V90 cohort and one-year-lagged EIA-860 thermal/nuclear/hydro/wind/solar capacities. No outcome-based additions or exclusions.
- Use factor sensitivity intervals (gCO₂/kWh): coal [900,1100], gas [360,440], oil [720,880], other fossil [200,600], nonthermal fuels 0. These are transparent stress bounds, not official China EF confidence intervals.
- Evaluate strict exceedance thresholds τ=600 and 900, with ≥1%, ≥5%, and ≥10% monthly tail targets.

## Risk-side information budget

- At τ=600, observe one monthly field `coal + oil MWh`; its entire EF interval is strictly above τ. The remaining `gas + other fossil` thermal MWh is inferred from the public-style thermal total and has EF in [200,600], hence is nonpositive for the strict event (including neutral EF=τ).
- At τ=900, observe one monthly field `coal MWh`; coal EF spans [900,1100], while all other thermal factors are at or below 880. The all-low coal endpoint gives K=0; the all-high coal endpoint uses the same one-field risk-side solver. The robust set is the endpoint hull across those monotone EF extremes.
- Keep monthly nonthermal group totals and annual technology capacities identical in both arms. Compare the risk-side set with the robust endpoint hull from all detailed fuel margins under the same capacities, the coarsest thermal-total budget, and no-capacity variants.
- For the detailed arm, compute sharp fixed-EF K sets at the all-low and all-high factor vectors. Coordinatewise monotonicity makes their endpoint hull an outer envelope for every independent EF vector in the box. For the side arm, use the exact two-side capacity solver with the stated EF ranges; verify endpoint containment of the detailed robust set and observed nominal proxy K.

## Required outputs and assertions

- Report paired row-level endpoints, widths, positive/negative/abstain decisions, and mean widths by threshold and target rate.
- Every reported information set must contain the observed nominal fixed-EF K. Detailed robust endpoints must lie inside the side-information robust endpoints; side sets must lie inside the coarsest monthly set.
- Report all failed capacity-feasibility rows rather than silently dropping them; preserve the frozen cohort denominator.

## Interpretation boundary

This is oracle coarsening on U.S. EIA hourly fuel data and fixed-factor proxy outcomes. It tests an information-compression theorem/mechanism, not China prediction skill, forecast-time availability, or empirical EF uncertainty. The cohort is retrospective and selected by V90's capacity audit.

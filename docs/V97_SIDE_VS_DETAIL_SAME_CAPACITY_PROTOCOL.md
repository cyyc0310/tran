# V97 protocol: pair threshold-side and detailed monthly margins under the same capacities

Frozen 2026-10-28. Re-evaluate the V96 threshold-side summary against detailed monthly fuel margins while holding annual capacity information exactly fixed. Use the same 358-region-month V90 cohort, τ=600/900, V54 EF proxy, lagged EIA-860 capacities aggregated to thermal/nuclear/hydro/wind/solar groups, and ≥1/5/10% risk targets.

For τ=600, the monthly threshold-side field is coal+oil MWh with side EF intervals high=[800,1000], low=400. For τ=900 it is coal MWh with high=1000, low=[400,800]. Compare detailed-month only vs threshold-side-month only; detailed-month + coarse technology capacity vs threshold-side-month + the same coarse technology capacity; and the fully coarse monthly and capacity budget.

Require rowwise actual-K containment, detailed-margin sets nested inside threshold-side sets, detailed-plus-coarse-capacity sets nested inside threshold-side-plus-coarse-capacity sets, and both side sets nested inside the coarsest budget. Retain the frozen retrospective cohort and do not filter by any additional result. This is an oracle coarsening and mechanism check, not China-data validation or forecast coverage.

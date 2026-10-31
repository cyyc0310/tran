# V90 protocol: coarse technology-group capacity and hourly tail identification

Frozen 2026-10-28. Question: do annual capacity envelopes at a coarse technology resolution still constrain monthly hourly exceedance counts when fuel-specific capacity is unavailable? This approximates a China-facing public-information budget (thermal, nuclear, hydro, wind, solar), but does **not** claim that an EIA-to-China schema has been validated.

## Frozen evaluation

- Use the same previously opened 25 EIA balancing authorities, 2024–2025 months, fixed EF proxy, thresholds 100/300, and event-frequency targets 1%, 5%, 10% as V89.
- Capacity vintage is one year lagged: 2023 for 2024 and 2024 for 2025.
- Aggregate EIA generator capacities into five shared hourly group caps: thermal = coal + gas + oil + other, plus nuclear, hydro, wind, solar. In each hour, all fuels in a group jointly use no more than the group's capacity. Monthly fuel totals retain the eight V54 category marginals. Multi-fuel generator capacity remains conservatively counted in every declared EIA source bucket before group aggregation.
- Retain months only if hourly coverage is at least 80% and observed group output peaks and group monthly energy satisfy the corresponding lagged capacity envelopes. This is retrospective mechanism screening because the audit uses target hourly labels; it cannot be presented as prospective availability or coverage.
- On the exact same retained months, also calculate a fuel-specific-capacity comparator from the same lagged inventory and the same optimization routine. Report that comparator only on the stricter subset where all observed fuel peaks and fuel totals satisfy the fuel-specific caps; show the paired coarse-vs-fine comparison separately from the larger coarse-only cohort.

## Identification calculation

For an event set of `k` hours among `T`, let `y_f` be fuel `f`'s total MWh in event hours, `M_f` its observed monthly total, `C_g` the cap of its coarse technology group, and `M_g` the group's monthly total. A necessary group allocation envelope is:

`0 <= y_f <= M_f`, and `max(0, M_g-(T-k)C_g) <= sum_{f in g} y_f <= min(M_g,k C_g)`.

These group constraints are also sufficient for group totals under the deliberately relaxed model: distribute each group's event and non-event totals evenly across its hours. Fuel composition can be allocated within the group to meet its monthly fuel marginals. No ramping, minimum-run, outages, imports, or storage constraints are imposed.

For each candidate `k`, the maximum event-group carbon excess is computed by maximizing a linear objective separately within each technology group. Within one group, fuels are sorted by their excess coefficient; the optimum lies at a group-total endpoint or the breakpoint where positive-coefficient fuels are exhausted. A candidate is retained if the maximum excess is positive and at least the monthly total excess `D` (up to a small outward floating-point tolerance). This relaxed group calculation can admit a zero-generation complement, so its candidate set is intersected with the exact V66 monthly-margin set. The resulting numerical set is conservative and nested by construction; it is not claimed to be the sharp capacity-constrained set. It must contain observed `K` or the run stops. The baseline and coarse-bound results are scored on the same coarse-capacity-feasible region-months.

## Interpretation limits and stop rules

- Report group-capacity feasibility rate, coverage, set widths, positive/negative/abstention rates, and all held-in region-months. Do not hide capacity-mapping failures from the quality audit.
- Do not compare coarse and fuel-specific capacity on different samples without separately reporting their intersection; any gain in usable months may come from coarser feasibility screening.
- Results are retrospective mechanism evidence on previously viewed EIA regions, not a frozen unseen-region confirmation, not China validation, and not actual CIF validation.
- A useful result must retain meaningful narrowing after fuel-specific capacities are removed. If the bounds collapse to the V66 set, annual coarse capacity alone is insufficient and the China-facing claim needs an additional local statistic or a weaker estimand.

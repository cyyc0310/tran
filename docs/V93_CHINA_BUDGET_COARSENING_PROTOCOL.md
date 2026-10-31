# V93 protocol: coarsen month fuel margins to a China-style generation table

Frozen 2026-10-28. V90 used detailed monthly EIA fuel margins even though the candidate Chinese information budget is coarser. V93 hides the within-thermal fuel shares and retains only monthly thermal, nuclear, hydro, wind, and solar generation totals plus one-year-lagged annual capacity by the same five technology classes. EIA hour-level fuel records are used only to construct these monthly coarse totals and to evaluate fixed-EF proxy K; this is an oracle coarsening experiment, not a China-data ingest or prospective forecast.

## Frozen cohort and comparisons

- Keep exactly the 358 region-month cohort in `results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv`; do not expand or re-screen it.
- Compare (a) coarse monthly generation categories alone, (b) the same coarse monthly categories plus the lagged technology-group capacities, (c) V90's detailed monthly fuel margins alone, and (d) V90 detailed monthly fuel margins plus capacities.
- Use the same 25 previously opened BA regions, 2024–2025 months, fixed EF proxy, thresholds 100/300, and ≥1/5/10% tail-count targets.
- For thermal, withhold coal/gas/oil/other monthly amounts and allow the event and complement blocks to choose any thermal EF in the V54 proxy support [400,1000]. Clean categories use EF 0 in this proxy. The fuel mix is free to differ between event and complement because no monthly fuel-specific totals are observed under this budget.
- Retain separate nuclear/hydro/wind/solar monthly totals and capacity limits when constructing their group-total allocation interval; they may be summed for the carbon objective because their proxy EF is identical.

## Feasibility calculation

For candidate `k`, aggregate thermal event generation `x` and clean event generation `y` each lie within intervals derived from the event and complement group capacity constraints. The event can be above threshold if the maximum event excess `Q_E=(1000−τ)x−τy` is positive. The complement can remain non-event if choosing the lowest thermal EF yields `Q_N=(400−τ)(M_T−x)−τ(M_C−y)<=0`. These are a rectangle intersected with one half-plane in `(x,y)`; enumerate its vertices, maximize `Q_E`, and check that some feasible point leaves positive complement generation. Because the feasible polygon is convex, an event-positive point and a positive-complement point can be mixed while preserving both strict conditions.

For no-capacity comparison, set each group's capacity to its monthly total, which makes the event and complement allocation intervals unconstrained for any interior `k`. Require actual K containment and capacity-set nesting within the coarse-month no-capacity set; otherwise stop. Do not compare against V90 on different samples.

## Limits

This proxy lets thermal fuel composition vary adversarially within the assumed EF range and across blocks. It therefore estimates the identified set under a broad coarse-data contract, not prediction performance. The 358-month cohort itself was chosen by retrospective target-hour capacity audits, labels are fixed-EF proxies, and no China hourly target or held-out region is used. A useful outcome is whether coarse public-generation fields plus annual capacity retain appreciable tail information after detailed fuel margins are removed.

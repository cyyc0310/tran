# V92 protocol: sharp K set under shared coarse technology capacity

Frozen 2026-10-28. Objective: remove the conservatism in V90's outward group-bound set by enforcing strictly positive generation in both the event-hour block and its complement. Compare the sharp set with V90's outer set on the exact same V90-retained region-months.

## Frozen data and scoring

- Use V90's 25 previously opened EIA balancing authorities, 2024–2025 months, one-year-lagged EIA-860 capacities, fixed EF proxy, thresholds 100/300, and 1/5/10% tail-frequency targets.
- Freeze the 358-region-month cohort in `results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv`. Do not re-screen or expand the cohort in V92.
- Keep V90 monthly-margin baseline and coarse outer set, and calculate a sharp grouped-capacity K set using the same fuel margins, group caps, hours, EF, and labels.

## Mathematical construction

For a given `K=k`, fuel `f` has known monthly amount `M_f`, event-block amount `y_f`, and excess coefficient `q_f=EF_f−τ`. For technology group `g`, total monthly amount is `M_g`, shared hourly capacity is `C_g`, and event-block total must satisfy

`L_g=max(0,M_g−(T−k)C_g) <= Σ_{f∈g} y_f <= U_g=min(M_g,kC_g)`, with `0<=y_f<=M_f`.

Those conditions are necessary and sufficient for group totals under the no-ramping, no-minimum-run, no-cross-hour-coupling model: distribute each event and complement fuel amount evenly across its respective hours. Since both block generation totals are required positive, each block then has positive generation at every assigned hour.

Let `D=Σ_f q_f M_f` be monthly excess. A valid interior split needs event excess `Σ q_f y_f>0` and complement excess `D−Σ q_f y_f<=0`. Therefore maximize event excess over the product of group allocation polytopes. Within each group, sort fuels by descending `q_f`; the maximum at fixed group total fills the highest-`q` fuels first. The group-total optimum is at a feasible interval endpoint or at the breakpoint where positive-`q` fuel mass is exhausted. The total maximum is the sum of group maxima.

If the maximum exceeds `max(0,D)`, a valid split exists. If the maximum equals positive `D`, the maximizing allocation must leave positive monthly generation in the complement; otherwise exclude it. K=0 and K=T are checked against the sharp monthly-margin set and monthly group-capacity feasibility. The resulting K set is sharp relative to these assumptions and decimal source values. It must contain observed K and be a subset of both V66 and V90 outer sets; otherwise stop.

## Interpretation limits

This is retrospective mechanism analysis. The frozen cohort was selected using target-hour capacity-peak and energy audits, and labels are fixed-EF EIA proxies rather than actual CIF. The theorem does not incorporate outages, imports, ramping, storage, minimum stable output, or uncertain/incorrect capacity mapping. It is not a China validation or forecast-coverage claim.

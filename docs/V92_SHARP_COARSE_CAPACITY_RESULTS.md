# V92: exact positive-block grouped-capacity set matches V90 on the retained sample

**Result: mathematical sharpening succeeded; empirical frontier did not improve.** V92 derived the exact K-feasibility condition under shared technology-group capacities while requiring positive generation in both the event and complement hour blocks. It was evaluated on the frozen 358 region-month V90 cohort, with no cohort expansion or re-screening.

## Derivation and checks

For a candidate count `k`, group event totals lie in `[max(0,M_g−(T−k)C_g), min(M_g,kC_g)]`. At a fixed event total, maximizing carbon excess fills fuels in descending `EF−τ` order. The group objective is piecewise linear, so its maximum is attained at an interval endpoint or where positive-excess fuel support is exhausted. Summing group maxima gives the maximum event excess `Q_max`.

An interior split is feasible when `Q_max>0` and `Q_max>=D`, where `D` is the monthly total excess. If equality holds, the maximizing split must leave positive generation in the complement. Since event excess is strictly positive, event generation is positive too. Equal within-block allocation then constructs positive-output hours with the required group caps and fixed fuel amounts.

The V92 script completed with all runtime checks passing:

- 358 region-months and 2,148 region-month × threshold × frequency rows were scored.
- Observed K was inside the exact set on all 2,148 rows.
- Exact sets were nested within V66 monthly-margin sets.
- Exact and V90 outer-set lower/upper endpoints were identical on all 2,148 rows; endpoint containment had no failures.

## Result: no interval-efficiency gain over V90

The exact feasibility condition produces the same reported intervals and certification frontier as V90 on this cohort. Region-equal macro mean widths remain about 419 hours at τ=100 and 442 hours at τ=300, compared with 544 hours for the monthly-margin baseline. At τ=100 and a ≥1% monthly-tail target, positive certification remains 75.8%; at τ=300 and ≥10%, it remains 30.7%, with 50.2% abstention.

This is useful because it closes the concern that the V90 result was an artifact of a loose zero-output complement. It does **not** create a new empirical gain or establish novelty by itself. The closed-form groupwise optimizer and exact feasibility theorem are likely a methods component; stronger novelty still requires realistic coarse monthly information, robust boundary/EF uncertainty, or blind China validation.

## Scope

The sharpness statement is conditional on fixed EF, known detailed monthly fuel margins, correct lagged shared group capacities, nonnegative output, and no ramping, outages, imports, storage, minimum-run, or other cross-hour restrictions. Although the capacity groups are coarse, V92 still uses EIA's detailed monthly fuel margins, which may be richer than the user's Chinese monthly tables. It remains retrospective EIA proxy evidence, not actual CIF or China deployment.

Protocol: [V92 protocol](V92_SHARP_COARSE_CAPACITY_PROTOCOL.md). Implementation: [V92 script](../scripts/v92_sharp_coarse_capacity_identification.py). Outputs: [`results/v92-sharp-coarse-capacity-r1`](../results/v92-sharp-coarse-capacity-r1/).

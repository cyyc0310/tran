# V345: monthly allocation ambiguity of omitted above-size scope output

Registered 2026-10-30 as a scope audit following V341–V343. The user identified the National Bureau of Statistics `fsMonthData` page as the direct provincial monthly query. Public interface documentation describes `fsMonthData` as provincial monthly data and the generation directory as offering current-period and cumulative series. The V345 calculation itself uses the locally archived OREO NBS-derived provincial CSV, not a direct API response; official values have not yet been compared row by row. The monthly source reports total generation and thermal, hydro, nuclear, wind and solar groups, but the thermal group is not fuel-resolved. The NBS-derived monthly series covers above-designated-size industrial enterprises; the matching annual series covers all plants. This experiment asks whether annual totals and public province capacity materially locate the omitted annual generation remainder within a month.

## Data and calculation

Use 2023 province monthly total generation, same-source annual all-plant total generation, and CEC provincial total installed capacity for 2022 and 2023. The archived dataset records February as a Jan–Feb cumulative bin and counts it once as 59 days; March–December use calendar-month lengths. A direct NBS API reproduction must request current-period and cumulative series separately, then confirm their semantics before constructing the same bins. Convert generation from GWh to MWh and CEC capacity from 10,000 kW units to MW.

For province `p`, let annual remainder `R_p = annual_all_plant_generation - sum(monthly_above_size_generation)`. For each reported period `m`, the capacity-only upper room is `U_pm = max(0, capacity_p × hours_m - observed_monthly_generation_pm)`. Given only `sum_m r_pm = R_p` and `0 ≤ r_pm ≤ U_pm`, compute the sharp per-period range:

`r_pm ∈ [max(0, R_p - sum_{j≠m} U_pj), min(R_p, U_pm)]`.

No technology is assigned to the remainder. Report 2022 and 2023 capacity vintages separately. The first is a sensitivity because one-year-lag capacity is not proven to upper-bound all 2023 capacity; 2023 is retrospective.

## Guardrails

This is a coarse feasibility calculation, not an hourly CIF bound or forecast. The result cannot establish the true monthly distribution of below-size output, its technology mix or emission factor. Do not treat the five reported technology categories as complete all-plant monthly generation. No capacity vintage is release-time validated. A positive result means only that annual totals plus this permissive capacity ceiling do or do not locate the remainder within the reported bins.

## Reproduction

`python scripts/v345_china_residual_monthly_scope.py`

Inputs, protocol, code and V320 source-audit summary are SHA-256 hashed into `results/v345-china-residual-monthly-scope-r1/summary.json`.

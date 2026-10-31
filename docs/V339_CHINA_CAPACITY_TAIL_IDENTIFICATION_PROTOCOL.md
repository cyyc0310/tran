# V339 protocol: monthly mix plus lagged capacity for Chinese tail-risk identification

Registered 2026-10-30. V338 found no violations of the necessary monthly-average generation/capacity condition on 2023 NBS-derived province-period technology totals when using lagged 2022 CEC capacity. V339 applies the group-capacity tail-count outer-set calculation from V90 to this Chinese information contract.

## Estimand and information

For each of 31 provinces and 11 NBS periods in 2023 (Jan–Feb cumulative plus March–December), identify the feasible number `K` of hours whose **generation-side direct CO₂ intensity within the NBS-reported generation pool** exceeds 100 or 300 g/kWh. This is a fixed-technology-factor scenario, not measured CIF and not consumption-side MEE EF. Monthly NBS generation totals are the observed margins. February represents 59 days; all other periods use actual calendar hours.

Technology groups are thermal, hydro, nuclear, wind and solar. For unknown thermal direct intensity, take the declared support `[400,1000]` g/kWh; all other technologies use zero direct operational CO₂ in this diagnostic. The interval over K is the union of feasible counts at the two support endpoints; because thermal remains above both thresholds and the other groups below, this is a conservative support envelope for these threshold comparisons. CEC's province-wide thermal/hydro/wind/solar installed capacities constrain the corresponding group; total installed capacity is the conservative nuclear upper bound because no separate provincial nuclear table is reported. Compare monthly margins alone with those same margins plus lagged 2022 capacity. Use 2023 year-end capacity only as a retrospective reference arm.

## Outcomes and audit

For each province-period report full K-set endpoints, widths, positive/negative/abstain decision for 1%, 5%, 10% tail frequency, and nestedness. Report all 341 province-periods; no hourly labels or target K are used to filter or score. Reconcile monthly category totals and preserve missing source entries as unknown/remainder, never zero. Require capacity energy-feasibility, interval nesting, and factor-support monotonicity across arms.

## Interpretation boundary

The result is an information-value/partial-identification frontier under explicit category, factor, denominator, and capacity assumptions. It is not a forecast-accuracy score, empirical coverage result, or sharpness claim about true Chinese hourly CIF. CEC capacity is retrospective; the 2023-vintage arm is ex-post. V338's monthly-average audit is necessary but does not verify each hour's capacity. An independent Chinese hourly generation/CIF label remains necessary to claim empirical coverage or prediction improvement.

# V341 complete-category China generation-risk hull

Registered 2026-10-30. User confirmed the National Bureau of Statistics monthly-data source reports thermal, hydro, nuclear, wind and solar generation; the public thermal field is not split into coal, gas, oil, waste heat or biomass. The NBS query page is JavaScript-rendered, so the query endpoint could not be read by the web text extractor. The official monthly energy-production releases and NBS energy reporting table independently list these five generation categories. The monthly figures describe above-designated-size industrial output and must not be treated as all generation.

## Corrections and inclusion

V339's capacity arms crossed 2022/2023 vintages and duplicated rows. V340 repaired those two defects, but its inclusion of Tianjin treated the absent hydro category as zero, and its nonnegative residual rule could not recover omitted hydro when known categories exceeded the published total. V341 excludes every province-period with a missing technology category. It reports those excluded records in the audit. It does not use the published total to impute a missing category or close rounding/scope discrepancies.

The analysis pool is the sum of all five reported technology categories for the 30 complete provinces and 11 2023 periods each. The published total is retained only for a scope/rounding discrepancy audit. It may differ from the category sum. The capacity arms are explicitly mapped: monthly only; monthly + 2022 year-end installed capacity; monthly + 2023 year-end capacity (retrospective).

## Estimand

For each complete province-period, compute an **outer hull** for K, the count of hours above 100 or 300 g/kWh under technology-level direct operational factors: thermal factor support `[400,1000]`; hydro, nuclear, wind and solar direct operational factor zero. Technology monthly totals are exact margins within the defined category pool. Thermal is not disaggregated by fuel, so this remains a scenario analysis, not measured CIF. CEC category capacity constrains thermal/hydro/wind/solar; the province total capacity is a loose upper bound for nuclear.

For the hull extrema over thermal-EF uncertainty, the scenario endpoints are used. For any fixed feasible hourly technology allocation, raising the thermal factor cannot reduce a count of hours over a fixed threshold; therefore the minimum hull endpoint is attained at the low support endpoint and the maximum at the high endpoint. The solver remains a conservative outer-hull routine: it does not enumerate the full integer K set or prove endpoint sharpness. Capacity is a necessary hourly nameplate bound only if its vintage and scope cover every asset in the monthly generation totals; this cannot be verified here. Therefore all results remain conditional mathematical restrictions, not empirical coverage or forecast skill.

Expected scoring rows: 30 provinces × 11 periods × 2 thresholds × 3 tail rates = 1,980. Expected monthly source audit rows: 31 × 11 = 341; the 11 periods with missing hydro are excluded.

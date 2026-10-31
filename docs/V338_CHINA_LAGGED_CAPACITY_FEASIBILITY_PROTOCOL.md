# V338 protocol: does lagged Chinese provincial capacity safely bound monthly generation?

Registered 2026-10-30 before scoring. V89/V90 found that annual capacity can tighten hourly tail-count sets in EIA fixed-factor proxies, but V90 explicitly required a Chinese category/boundary/vintage audit. CEC annual capacity by province and technology is now transcribed for 2022–2023. The yearbook reports total, thermal, hydro, wind, and solar capacities, but no province-level nuclear capacity table; total installed capacity is therefore used as a conservative upper bound for nuclear. V320's NBS-derived monthly production gives a matching five-technology generation schema.

## Scope and test

This is a necessary-condition feasibility audit, not a CIF forecast or interval-coverage test. Match 2023 province-month NBS generation categories (thermal, hydro, nuclear, wind, solar) to CEC provincial installed capacity for (a) the lagged 2022 vintage and (b) contemporaneous year-end 2023 capacity. NBS February is a Jan–Feb cumulative observation, so use 59 days for its energy/capacity denominator; March–December use calendar-month hours. Capacity is reported in 10,000 kW; convert 1 unit to 10 MW.

For each available province-period-technology, calculate the period-average capacity factor `generation_MWh / (capacity_MW × period_hours)`. A value above 1 violates the claimed hard upper bound even before hourly peaks are examined. Report all groups, not just months passing the bound; no filtering by hourly labels. Missing technology reports are recorded as unavailable, not imputed to zero. Audit CEC category sum against total and source-wise national sums.

## Decision

If one-year-lag CEC capacity frequently violates even monthly average output, stop using it as a hard China bound without a prospective capacity-growth envelope. If contemporaneous capacity passes, it remains retrospective and cannot support an as-of forecast. Passing this test is necessary but not sufficient: an independent Chinese hourly generation/CIF label would still be required to verify tail-set containment and risk coverage.

## Sources and caveats

CEC *China Electric Power Statistical Yearbook 2024*, tables 2-1, 2-3, 2-5, 2-7, 2-9; source PDF SHA-256 `6aebb42952155995cc43041de542516d8a37b9171db5d74575ea261d76bcf15c`. NBS monthly generation is the above-designated-size statistical series audited in V320; CEC capacity is province-wide. The yearbook is retrospective and its release/vintage is not treated as available at the 2023 forecast origin. Province-wide installed capacity is intentionally a loose upper bound for the smaller NBS generation reporting universe. Capacity categories do not exactly sum to total in a few provinces; the signed difference is retained as an audit field and is not assigned to nuclear.

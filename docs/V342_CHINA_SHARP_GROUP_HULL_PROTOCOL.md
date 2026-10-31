# V342 exact fixed-factor K set and hull tightness audit

Registered 2026-10-30 after V341's complete-case outer-hull frontier. Reuse the identical 330 province-period cohort, category margins, CEC 2022/2023 capacities, EF support, threshold, and denominator. Do not use hourly China labels (none are present in this audit).

## Question

How conservative is the V341 outer hull compared with the exact feasible integer K set under the same group-capacity contract? Implement a separate exact fixed-factor set solver for the five single-category groups, require positive generation in both event and non-event hours, and compare monthly-only and each capacity vintage. For each EF support endpoint (thermal 400 and 1000 g/kWh; all other direct operational factors zero), return the full feasible K set. For the interval over thermal EF, report only the safe hull extrema: monotonicity of each fixed hourly allocation implies the minimum K over the EF support occurs at 400 and the maximum occurs at 1000. Do not treat the union of endpoint sets as the complete feasible set under continuous EF support.

The monthly-only arm uses a deliberately nonbinding per-group capacity (each group's entire month total could fit in one hour). Capacity arms use the exact V341 2022 and 2023 CEC mappings. Check energy-capacity feasibility before all K shortcuts. Require exact V341 cohort keys, output full fixed-factor K sets, hull endpoints, width, and gap count. No benchmark promotion unless exact capacity bounds materially improve efficiency over V341 and later truth containment is verified on independent hourly data.

## Interpretation boundary

The fixed-factor group-capacity exact-set construction already exists in the project's EIA-oriented V92/V102 line. This experiment tests tightness and transfer of that established mathematical tool to China's coarser five-category, uncertain-thermal-EF information contract. It is not by itself an algorithmic novelty claim, real-CIF validation, or forecast result.

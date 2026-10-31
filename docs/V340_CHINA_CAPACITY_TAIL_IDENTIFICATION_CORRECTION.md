# V340 corrected China capacity tail identification

Registered 2026-10-30 after discovering a vintage-labeling and duplication defect in V339. V339's original script and outputs are preserved, but its claimed comparison between lagged 2022 and same-year 2023 capacity is invalid: inside each capacity-vintage loop both named arms received the currently iterated capacity, rows were emitted twice, and summary denominators were reported as if they were unique province-periods.

V340 emits exactly one record for each 31 provinces × 11 published periods × 2 thresholds × 3 tail rates (2,046 rows). The monthly-only arm uses no capacity; the lagged arm always uses CEC 2022 capacity; the retrospective arm always uses CEC 2023 capacity. It checks category-energy feasibility against both capacity vintages and preserves monthly category/total discrepancies and Tianjin's missing hydro entry as a residual/unknown group.

The estimand remains the conditional outer identified set for the number of hours above 100 or 300 g/kWh under fixed technology-factor scenarios: thermal `[400,1000]`, nonthermal direct generation emissions zero, and unresolved group `[0,1000]`. CEC capacity is retrospective; it is not available as a validated as-of predictor in this dataset. No Chinese hourly generation or CIF labels are available here. Thus the results can assess mathematical information value under stated assumptions, but cannot claim empirical coverage, forecast skill, or sharp identified bounds.

The correction does not change the previous failure interpretation: China monthly energy alone cannot resolve hour-level tail frequency. It specifically reopens only the capacity-vintage comparison and exact denominator after correcting V339.

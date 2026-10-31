# V325 protocol: lagged intensity-contrast route selection

Registered 2026-10-30 before the V325 LP run. V324 found that top-volume sparse route-month measurements approached the worst-case value of province receipt totals, but did not clearly outperform them. V325 tests whether a simple, interpretable route-selection rule using lagged carbon-intensity contrast improves value per field.

## Question

Under the same 310-field budget, do monthly measurements on routes with the largest expected source-versus-receiver intensity contrast improve worst-case import-exposure bounds over (a) the 31 highest annual OD-volume routes and (b) all 31 receiver monthly totals?

## Frozen selection rule and inputs

Use the same V321/V324 feasible set, target, NBS 2023 monthly five-category supply caps, 2023 annual V310 OD matrix, Ember source factors for the 2023 target, and residual-factor support `[0,1.0] kgCO₂e/kWh`. For **route selection only**, calculate each province's lagged 2022 annual generation-weighted emissions intensity from Ember's 2022 technology-generation and emissions estimates. Rank each positive OD edge by `2023 annual OD flow × absolute(2022 origin intensity − 2022 destination intensity)`, descending; break ties by origin and destination alphabetically. Select 31 routes and observe the first 10 reporting bins on each, for 310 independent route-month fields; each route's final bin follows from its annual total. The rule may not use 2023 target-period intensity or LP width when selecting routes.

Compare the frozen contrast-ranked set, V324's top-volume-ranked set, and 310 province receipt totals. Also evaluate receipts combined with each route design at an explicitly doubled 620-field budget. Cross total-only and five-category monthly supply caps. For each design, a two-copy LP maximizes the conditional width over all possible shared measurement values.

## Primary outcomes

For the five-category supply arm, report mean and median paired reduction, positive cells and provinces, and province-cluster bootstrap intervals for the contrast-route minus top-volume route and contrast-route minus receiver-total differences. Check that additional fields never widen bounds. The estimand remains a direct-import exposure proxy divided by local above-size generation, not consumption CIF or hourly risk. This is one 2023 proxy case and does not validate the selection rule across years; no target-outcome tuning is permitted.

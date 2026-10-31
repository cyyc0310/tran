# V326 protocol: receiver-balanced sparse route selection

Registered 2026-10-30 before the V326 LP run. V324/V325 found that sparse route fields produced gains in relatively few provinces; V325's lagged-intensity contrast heuristic did not improve over simple annual-volume ranking. V326 tests destination coverage as a distinct design principle.

## Frozen route-selection rule

Using only the 2023 V310 annual OD matrix, select for each of the 31 destinations the single incoming route with the largest annual volume. The matrix has at least one incoming edge for every province; therefore this selects exactly 31 distinct routes. Observe the first 10 reporting bins on each selected route (310 independent monthly route values); the final bin follows from each known annual route total. Do not tune based on target bounds. Compare with V324's 31 globally largest routes and 310 receiver-month totals.

## Fixed estimand and feasible set

Reuse V321/V324 inputs and assumptions: 2023 NBS-derived monthly generation by five categories, annual all-plant remainder, 2023 annual OD, Ember monthly source-category factors, missing-category handling, residual EF support `[0,1.0] kgCO₂e/kWh`, and 11 bins (Jan-Feb plus March–December). The target is source-weighted direct imports divided by local reported generation, not consumption CIF or hourly risk. For each field set, a two-copy LP maximizes the conditional diameter over all possible shared measurement values. Cross total-only and five-category supply constraints; evaluate 620-field combinations only as a separate complementarity arm.

## Primary evaluation

Report mean/median paired width reduction, positive province-period cells, positive provinces, and province-cluster bootstrap intervals for receiver-balanced routes versus top-volume routes and receiver totals. Verify additional measurements never widen any bound. This is one-year design evidence; a gain is a candidate result requiring transfer tests, not a confirmed contribution.

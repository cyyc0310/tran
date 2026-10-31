# V324 protocol: sparse route-month versus receiver-month field value

Registered 2026-10-30 before the V324 LP run. This extends the V321/V323 scope-aware feasible set and compares two physically meaningful measurement designs with the same budget.

## Question

With 310 independent monthly fields, does observing the monthly receiving total for every province or monthly flows on a small set of paired origin-destination routes yield narrower worst-case direct-import exposure bounds?

## Fixed inputs and assumptions

Use the V321 data contract and feasible set: 2023 NBS-derived monthly above-designated-size generation by five categories, annual all-plant remainder, V310 annual OD, Ember source-category monthly emission-factor proxies, missing-category handling, and residual factor support `[0,1.0] kgCO₂e/kWh`. The residual support is a sensitivity scenario, not a validated engineering bound. Keep 11 reporting bins (Jan-Feb and March–December), 31 provinces, 116 annual OD routes, and the same local-generation denominator.

## Equal-budget measurement designs

1. **Receiver totals:** 31 provinces × 10 independent monthly receipt totals; the 11th period follows from each province's annual OD receipts.
2. **Sparse paired routes:** select the 31 largest routes by fixed annual OD volume, breaking ties alphabetically by origin then destination; observe each route's first 10 reporting-bin flows. The 11th follows from its known annual route total. This is 310 independent measurements.
3. **Combined:** both designs (620 fields), used only to measure complementarity.

For each information set, use two feasible copies with identical released-field values and maximize the target diameter over all feasible releases. This is a worst-case conditional-diameter criterion, not expected value under a fitted distribution. Cross total-only and five-category monthly supply caps. Preserve the predeclared top-31 route selection; do not tune route selection on target outcomes.

## Primary comparisons

Compare the 310-field route design with the 310-field receiver design for the five-category supply arm: mean and median paired reduction, positive province-period cells, positive province averages, and province-cluster bootstrap 95% interval for the paired difference. Confirm adding measurements cannot widen bounds. The endpoint is an imported source-weighted emissions proxy divided by local reported generation, not consumption CIF and not an hourly risk label.

This is a structural information-design experiment. Even a positive result would not establish that the selected flow measurements are public, available in China in the required accounting definition, or predictive of hourly CIF. A null result would prioritize broader receiving totals over a concentrated set of high-volume OD routes under this feasible set.

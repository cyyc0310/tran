# V346: propagate the monthly scope remainder into hourly tail-count bounds

Registered 2026-10-30 following V345. Add the NBS monthly reporting-scope remainder to the five-category generation-side hourly K identification model. Use V341's 330 complete province-periods, V342's exact fixed-margin solver as the baseline, and V345's 2023-capacity feasible monthly remainder range. Thresholds are 100 and 300 g/kWh.

## Information and model

Known reported generation margins remain thermal, hydro, nuclear, wind and solar. Thermal EF support is [400,1000] g/kWh and other reported categories have zero direct operational EF. Add an unresolved generation group whose monthly mass varies over the V345 feasible interval and whose EF support is [0,1000] g/kWh. Since K is monotone in each group's EF for a fixed allocation, compute the lower hull at thermal EF 400/residual EF 0 and the upper hull at thermal EF 1000/residual EF 1000.

Retain V342's independent group-capacity convention for observed groups. For capacity arms, use the same vintage's province total capacity for the unresolved hourly cap and the matching V345 residual range (2022 for lagged, 2023 for retrospective); for the monthly-only arm, permit the full annual remainder to fit in one hour. These caps can allow simultaneous generation above total system capacity, so the resulting hull is a conservative outer relaxation. Treat each province-period remainder range cellwise; do not claim a joint annual allocation model. The V345 finding that the residual can be zero in any target period means the new hull should contain the baseline hull.

## Method and checks

For the outer hull endpoints, evaluate the zero-residual and maximum-feasible-residual masses at the EF support endpoints. At fixed EF, additional zero-EF generation can only lower hourly intensity, while additional high-EF generation can only raise it; the extremal K counts therefore occur at the maximum residual mass or at the zero-mass baseline. The factors are monotone as well. Retain the minimum/maximum K over these cases and compare with V342. Reject any result that violates baseline-set nesting or the expected 330 × 2 × 3 key count. This is a hull calculation; it does not claim that the union of endpoint integer sets fills every K inside the hull.

## Limits

This is a model-conditional direct-generation tail-count outer hull, not consumption-side CIF, an empirical prediction interval, or a coverage result. The residual factor support is a sensitivity support, not a measured Chinese factor range. Independent technology caps permit simultaneous output above total capacity and make the result conservative. V345 inputs are NBS-derived local archives; direct official API values and revision vintages remain to be checked. No held-out hourly Chinese labels are used.

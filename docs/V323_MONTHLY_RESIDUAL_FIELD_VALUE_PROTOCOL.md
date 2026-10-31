# V323 protocol: minimax value of monthly residual-generation measurements

Registered 2026-10-30 before the V323 LP run. This extends V321 with a hypothetical but physically interpretable measurement: the month/bin allocation of each province's already-known annual below-designated-size generation remainder.

## Question

Under V321's scope-aware feasible set, does releasing 310 independent province-period residual-generation amounts (31 provinces × 10 periods; the 11th follows the annual residual) shrink the worst-case direct-import exposure diameter more than 310 province-period receipt totals?

## Inputs and fixed assumptions

Use unchanged V321 inputs, source factors, period bins (`Jan-Feb`, March–December), annual OD, monthly NBS category supply, annual all-plant NBS totals, missing-Tianjin-hydro treatment, and residual factor support `[0,1.0] kgCO2e/kWh`. Do not alter the annual residual amount. No profile is filled from Ember or any other estimated monthly shape.

## Measurement-set construction

In the no-residual-profile arm, retain V321's annual-only residual constraint: monthly unobserved exports may be allocated arbitrarily, subject to the province's annual residual supply.

In the measured-residual-profile arm, introduce a shared latent field `r[i,m]` across the two feasible copies, with `r[i,m] ≥ 0` and `sum_m r[i,m] = annual residual[i]`. Each copy's unobserved exports from province `i` in period `m` are capped by this same `r[i,m]`. The two-copy LP jointly maximizes over all possible observed profiles, yielding the **worst conditional diameter over any possible released profile**, not a chosen or oracle profile. The 11th period is implied by the annual total, giving 310 independent measurements.

Keep the equal-budget receipt arm (310 destination-period totals) and a combined 620-field arm. Cross total-only versus five-category observed monthly generation caps. Use the same two-copy LP, target denominator, and EF support as V321.

## Primary comparisons

- Add residual profile (310 fields) versus no added fields.
- Add monthly receipt totals (310 fields) versus no added fields.
- Compare both 310-field schemes on the same province-period targets.
- Add both (620 fields) only as an explicit combined-information arm.
- Report pooled and paired relative diameters, medians, positive cells, and province-cluster bootstrap intervals.

This is a minimax information-value experiment, not a claim that the monthly residual dataset is currently public or available to the user. The endpoint remains an import-exposure proxy, not consumption CIF or hourly risk. If the `r` profile has negligible value, the missing minimum information is more likely source/receiver pairing or emissions-intensity information; if it sharply reduces widths, the next data target is full-scope monthly generation by province.

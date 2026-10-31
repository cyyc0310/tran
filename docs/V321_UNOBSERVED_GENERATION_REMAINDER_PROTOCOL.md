# V321 protocol: explicit unobserved-generation remainder in the Chinese monthly-flow bounds

Registered 2026-10-30 after V320's scope audit, before running the V321 LP.

## Question

Does V319's monthly-category information value survive after representing (i) generation below the NBS monthly above-designated-size reporting scope, and (ii) the missing monthly hydro category for Tianjin, as unobserved rather than zero?

## Frozen input contract

- OREO/NBS 2023 monthly province-generation fields: total, thermal, hydro, nuclear, wind, solar. February is the January–February cumulative period. Analyze 11 reporting periods: `Jan–Feb`, March, …, December.
- OREO/NBS 2023 annual all-plant generation by province. Per-province residual pool is annual total generation minus the sum of the monthly reported total, with no negative clipping permitted. A negative residual or OD infeasibility is a reported data-contract failure.
- V310 2023 annual OD matrix remains a separate, unverified accounting source.
- Ember 2023 source/month intensity is used only as an operational factor proxy for observed monthly categories. For `Jan–Feb`, compute emissions divided by generation after summing January and February.
- The unobserved remainder's operational factor lies in `[0, 1.0] kgCO2e/kWh`. `1.0` is the maximum 2023 source-month intensity in the Ember proxy, not a validated engineering bound; treat this as an explicit scenario and report its consequence.

## Feasible set and information comparison

Use route × category × 11-period nonnegative energy variables. Every route's 11-period/category totals equal its annual OD value. All reported above-size source flows are capped by the NBS monthly total; the five-category arm additionally applies category-specific NBS monthly caps. A missing category is not set to zero: its difference from the reported monthly total becomes a within-scope unclassified pool. The annual below-size pool may be allocated across any period, but its province-wide total cannot exceed the all-plant annual residual. Unobserved-pool carbon mass is represented linearly by `0 ≤ carbon_mass ≤ 1.0 × unobserved_energy`.

For each of 31 receivers × 11 periods, compute the two-copy worst-case conditional interval diameter. Compare total-only versus five-category monthly fields, each with and without 310 hypothetical independent monthly receipt totals (31 destinations × 10 periods; the final period follows annual OD receipts). The target denominator is NBS monthly above-size local generation for that period; it remains an import-exposure proxy, not consumption CIF.

## Interpretation rules

- If adding the residual pool destroys most category-field shrinkage, the minimum local-data requirement includes either full-coverage monthly generation or a validated residual-generation allocation by time and technology.
- If category fields remain informative, the gain is conditional on the `[0,1.0]` operational-factor support and the V310 OD matrix.
- No result from this experiment establishes forecast coverage, real flow attribution, or hourly CIF validity. Do not use V64 modeled load here.
- Do not use user-held generation data as verified until its original provider, plant coverage, February/cumulative convention, units, revision history, and category mapping are confirmed.

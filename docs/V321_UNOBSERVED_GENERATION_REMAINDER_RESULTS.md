# V321 results: incomplete monthly-generation coverage erodes category-field value

Run date: 2026-10-30. Frozen protocol: [V321](V321_UNOBSERVED_GENERATION_REMAINDER_PROTOCOL.md). All 341 province-period queries completed in four information arms. Full results and hashes: `results/v321-unobserved-generation-remainder-r1/`.

## Correcting the monthly coverage gap

The model used 11 NBS reporting periods (`Jan-Feb`, March–December), 116 annual OD routes, and an annual unobserved-supply pool of **601.58 TWh**, equal to the same-source 2023 all-plant generation minus the published monthly above-size generation. For Tianjin, monthly hydro was absent in all 11 periods; the model left the hydro cap uninformative up to the reported total rather than setting it to zero. Missing below-size generation could be allocated to any period subject only to its annual provincial remainder. Its emissions factor varied from 0 to 1.0 kgCO₂e/kWh; the upper value is a proxy support taken from the largest 2023 Ember source-month factor, not an engineering guarantee.

## Results

| Information set | Added receipt fields | Mean diameter | Median diameter |
|---|---:|---:|---:|
| Monthly total + annual residual | 0 | 3.189 | 1.948 |
| Five categories + annual residual | 0 | 3.038 | 1.735 |
| Monthly total + annual residual | 310 | 3.189 | 1.948 |
| Five categories + annual residual | 310 | 2.658 | 1.505 |

Across matched province-period cells, five-category fields reduced the total-only diameter by **3.62%** on average when no receipts were added (median 0%; 131/341 cells narrowed; province-cluster bootstrap 95% interval 1.81%–5.76%). Only 13/31 provinces had a positive average reduction. With 310 hypothetical monthly receipt fields, the five-category advantage rose to **11.20%** (median 0.30%; 187/341 cells narrowed; 17/31 provinces positive; bootstrap 95% interval 6.47%–16.25%). The receipt fields alone changed the total-only diameter by less than 0.001% on average. Conditional on five-category caps, receipts reduced diameter 8.03% on average, but the median reduction was still 0.30%.

The no-receipt effect is substantially smaller than V319's 21.5% because V321 does not treat the 6.36% annual generation remainder as absent. The V319 figure cannot be carried over to the NBS data contract. In V321, category resolution alone helps a subset of province-periods; its value appears mainly in combination with recipient-month information. The effect is uneven: Guangdong had the largest average category-only reduction (25.4%), while many province-period cells did not narrow.

## Interpretation and decision

This is the first quantitative result under the documented NBS monthly reporting scope. It provides a candidate **scope-aware information frontier**: monthly generation categories plus a separately declared annual residual, with optional recipient-month totals. It also shows why silently assuming full monthly coverage can overstate a field's usefulness.

The result is not standard CIF, because the denominator is reported above-size local generation rather than consumption; Jan-Feb is a two-month bin; route flows come from the unverified V310 source; category carbon factors use Ember estimates; and the residual factor interval is an explicit scenario. The LP does not validate interval coverage for hourly predictions. A paper contribution could be a rigorous, scope-aware identifiability benchmark, but no original forecasting algorithm breakthrough is established.

V322 completed a predeclared sensitivity over the unobserved residual EF upper endpoint (0.5, 1.0, and 1.5 kgCO₂e/kWh). Average category-field gains remained positive across scenarios, but the median no-receipt gain fell to zero at upper endpoints 1.0 and 1.5. The paper should therefore report an uncertainty frontier over the factor support, not one best-case shrinkage percentage; see [V322 results](V322_RESIDUAL_FACTOR_SUPPORT_SENSITIVITY_RESULTS.md).

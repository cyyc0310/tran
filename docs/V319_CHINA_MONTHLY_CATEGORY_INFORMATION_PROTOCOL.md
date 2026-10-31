# V319 protocol: origin-corrected value of Chinese monthly generation categories

Registered 2026-10-30 before the V319 run. This is a structural partial-identification experiment, not an hourly-CIF forecast.

## Audit correction

V315/V316 used the receiving province's technology-specific intensity in the incoming-flow objective. For an imported MWh generated in province `i`, the source factor must be `e[i,k,m]`, not `e[j,k,m]`. The new script uses the origin-resolved factor throughout. Prior V315/V316 values are superseded for interpretation and must not be cited as valid incoming-carbon-risk bounds.

## Question

Under the same annual OD matrix, source-month generation caps, two-copy LP objective, and 341-field receipt budget, does monthly generation composition still reduce the worst-case interval diameter when the eight Ember source rows are aggregated to a five-category public-report schema: thermal (coal, gas, bioenergy, other fossil), hydro, nuclear, wind, and solar?

## Frozen design

- Target year: 2023; 31 provinces; 12 months; 116 positive annual OD routes.
- Base variables: nonnegative `x[i,j,k,m]`, where `k` is one of eight detailed Ember categories and `m` is month.
- Annual-route conservation: `sum(k,m) x[i,j,k,m] = OD[i,j]`.
- Supply caps compared: (a) monthly total generation, (b) five grouped monthly categories, (c) eight detailed categories. The eight-category variables remain in all arms; coarse grouping changes only what is observed/capped.
- Optional equal-budget field set: monthly receipt totals for each destination for 11 months (31×11 = 341 independent numbers; month 12 follows annual OD totals).
- Source factor: same-month, same-source-province `Emissions / Generation` from the Ember snapshot, `e[i,k,m]`.
- Target: incoming source-weighted emissions divided by receiver's monthly total generation, retained only as a direct-import exposure proxy.
- For each receiver-month and information set, compute worst-case conditional diameter with two feasible flow matrices constrained to agree on all published fields.
- Report mean/median diameter and paired per-cell percent shrinkage relative to each supply-cap arm without receipt fields. Preserve infeasibility/errors; do not force feasibility.

## Limitations and interpretation

The annual OD source and Ember 2023 monthly generation/emissions are retrospective secondary datasets. Generation caps are not observed exports; annual OD and monthly supply may have incompatible scopes; the denominator is not consumption; no hourly flow, load, losses, or independent CIF label is available. This experiment can evaluate the mathematical value of coarse category fields inside this declared feasible set only. It cannot validate real Chinese CIF or hourly prediction. The “thermal” grouping is a schema proxy and does not assert that every official monthly table has identical coverage or definitions.

## Execution

`python scripts/v319_china_monthly_category_information.py`

Outputs are written once to `results/v319-china-monthly-category-information-r1/`.

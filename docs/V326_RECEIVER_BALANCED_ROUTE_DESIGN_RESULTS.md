# V326 results: destination-balanced sparse OD monitoring

**Run date:** 2026-10-30  
**Protocol:** [V326 protocol](V326_RECEIVER_BALANCED_ROUTE_DESIGN_PROTOCOL.md)  
**Reproduction:** `python scripts/v326_receiver_balanced_route_design.py`  
**Machine-readable output:** `results/v326-receiver-balanced-route-design-r1/summary.json` and `province_period_diameters.csv`

## Question and design

V324/V325 showed sparse route measurements often helped a minority of provinces. V326 fixes that coverage pattern by selecting one largest incoming annual route for each destination. Since all 31 provinces have at least one nonzero incoming V310 route, the rule selects exactly 31 distinct routes and 10 independent reporting-bin flows per route, matching the 310-field receipt and global top-volume designs.

The receiver-balanced route set carries 55.15% of total annual OD volume, compared with 77.56% for the globally largest 31 routes. The model and target are otherwise unchanged from V321/V324: 2023 NBS-derived five-category monthly supply caps, annual remainder, V310 annual OD, Ember source factors, residual EF support `[0,1] kgCO₂e/kWh`, and local reported generation as denominator. The target is a direct-import exposure proxy, not consumption CIF or hourly risk.

## Results

| Five-category supply information | Added fields | Mean diameter | Median diameter | Mean paired reduction vs none | Median reduction | Positive cells | Positive provinces |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 0 | 3.0384 | 1.7353 | — | — | — | — |
| Global top-31 routes | 310 | 2.6613 | 1.4592 | 7.24% | 0.00% | 154/341 | 14/31 |
| One largest route per receiver | 310 | 2.7388 | 1.5046 | 6.83% | 0.30% | 187/341 | 17/31 |
| Receiver monthly totals | 310 | 2.6584 | 1.5046 | 8.03% | 0.30% | 187/341 | 17/31 |

The receiver-balanced route design spreads positive gains to the same number of province-period cells and provinces as receiver totals, while lowering mean reduction by 1.20 percentage points. Its paired mean difference versus global top-volume routes is −0.41 points (province-cluster bootstrap 95% interval −3.09 to +2.31); versus receiver totals it is −1.20 points (−2.94 to +0.03). The balanced route design therefore improves the *distribution* of gains relative to global top-volume selection but does not beat receipt totals in average width. The confidence intervals do not establish a clear pairwise advantage.

Combining balanced routes and receiver totals at a doubled 620-field budget yields 8.57% mean reduction, below volume routes plus receipts at 9.11%; the median is 0.30% for both. As in prior rounds, all added-field schemes have essentially zero value under total-only supply caps. Monotonicity audits pass.

## Interpretation and next direction

Receiver coverage is a useful design constraint: it increases the count of helped province-periods from 154 to 187 relative to globally ranking routes by volume, while sacrificing some mean reduction. But the gain pattern exactly matches receipt totals on coverage counts, and the route design does not outperform them. This is an interpretable trade-off, not a paper-level breakthrough.

Three route-selection heuristics have now been tested—annual volume, lagged source-receiver intensity contrast, and destination-balanced maximum-incoming route—and none improves the overall worst-case width over receiver totals. Stop tuning route rankings. The next research move should leave hypothetical annual-OD information design and return to a labeled temporal forecasting experiment: test whether locally available monthly fuel-generation categories plus a lagged annual level anchor improve hourly distribution/threshold-risk prediction on genuinely unseen regions, against unanchored, local-persistence, seasonal, and conformal baselines. A positive claim requires independent hourly labels and coverage-efficiency gains; otherwise position the contribution as an information-identifiability benchmark with explicit data-boundary limits.

## Limitations

- OD routes and monthly generation are from separate sources and are not reconciled.
- The EF sources and residual support are proxy assumptions; support sensitivity is substantial in V322.
- Monthly NBS-derived generation covers above-designated-size enterprises, and the annual remainder is not monthly resolved.
- This optimizes worst-case conditional diameters for hypothetical measurement values, not expected empirical error reduction.
- One OD year cannot validate cross-year route selection or forecasting transfer.

The exact 31 selected routes, paired cell widths, bootstrap outputs, source hashes, and monotonicity audit are in `summary.json` and `province_period_diameters.csv`.

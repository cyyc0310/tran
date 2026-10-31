# V325 results: lagged intensity-contrast route selection

**Run date:** 2026-10-30  
**Protocol:** [V325 protocol](V325_CONTRAST_RANKED_ROUTE_DESIGN_PROTOCOL.md)  
**Reproduction:** `python scripts/v325_contrast_ranked_route_design.py`  
**Machine-readable output:** `results/v325-contrast-ranked-route-design-r1/summary.json` and `province_period_diameters.csv`

## Question and design

V324 found that 31 highest-volume route-month measurements were close to province receipt totals under a 310-field budget. V325 tested whether a more targeted frozen heuristic improves route selection: rank each route by 2023 annual OD volume multiplied by the absolute difference in origin and destination annual generation-weighted emissions intensity from 2022 Ember data. Select the top 31 and observe 10 bins per route. This avoids using 2023 target EF values for route ranking. The target LP remains the V321/V324 two-copy minimax direct-import exposure proxy, with 2023 Ember source-category factors and residual EF support `[0,1] kgCO₂e/kWh`.

## Results

| Added field set, five-category supply | Fields | Mean diameter | Median diameter | Mean paired reduction | Median reduction |
|---|---:|---:|---:|---:|---:|
| None | 0 | 3.0384 | 1.7353 | — | — |
| Top-31 annual-volume routes | 310 | 2.6613 | 1.4592 | 7.24% | 0.00% |
| Top-31 lagged-intensity-contrast routes | 310 | 2.6945 | 1.6360 | 5.91% | 0.00% |
| Province receiver totals | 310 | 2.6584 | 1.5046 | 8.03% | 0.30% |

The contrast-ranked routes are positive in 143/341 province-period cells and in 13/31 province averages, compared with 154/341 and 14/31 for volume-ranked routes, and 187/341 and 17/31 for receiver totals. The contrast-minus-volume mean difference is −1.33 percentage points (province-cluster bootstrap 95% interval −4.62 to +1.22); contrast minus receipts is −2.12 points (−5.71 to +0.42). Neither paired interval excludes zero, but both differences favor the simpler baselines in point estimates. The contrast-ranked routes cover 65.04% of annual OD volume, versus 77.56% for the top-volume set. Combining contrast routes with receipts at twice the budget yields 9.11% mean reduction, effectively the same as volume routes plus receipts (9.11%).

As in V323/V324, all new-field arms have essentially zero value under total-generation-only supply caps. Gains require the five-category source-side restrictions. Bound monotonicity holds.

## Interpretation

The tested scalar carbon-contrast score does not improve sparse monitoring. It appears to trade away too much OD volume for route-factor contrast and does not broaden the province coverage enough. Do not promote it as a method contribution. The more promising next diagnostic is a receiver-stratified route set: select at least one feasible route per destination, then fill the remaining 31-route budget by annual flow. This tests coverage of destination constraints directly and avoids tuning on target LP outcomes. If it also fails to improve the width/coverage distribution, the evidence favors receiver totals as the operationally simpler field and leaves the partial-identification benchmark—not route selection—as the main research candidate.

## Limitations

- This is one OD year and one proxy factor system; it does not establish cross-year transfer of the route-selection rule.
- V310 OD and NBS-derived generation remain cross-source and unreconciled.
- The residual EF support is a declared proxy interval, not an engineering bound; V322 documents support sensitivity.
- The target is local-generation-normalized direct-import exposure, not consumption CIF or hourly risk.
- Minimax widths optimize over all possible measurement releases; they are not expected empirical gains.
- The route score is one frozen heuristic, not an optimal design search.

Full route rankings, paired bootstrap output, input hashes, and cellwise monotonicity are recorded in `summary.json`.

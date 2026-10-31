# V324 results: sparse paired OD flows versus receiver totals

**Run date:** 2026-10-30  
**Protocol:** [V324 protocol](V324_SPARSE_MONTHLY_OD_VALUE_PROTOCOL.md)  
**Reproduction:** `python scripts/v324_sparse_monthly_od_value.py`  
**Machine-readable output:** `results/v324-sparse-monthly-od-value-r1/summary.json` and `province_period_diameters.csv`

## Question and setup

V323 found that 310 hypothetical monthly receiver totals reduced bounds in the five-category supply arm, while monthly allocation of the unobserved annual generation remainder had no worst-case value. V324 asks whether equally many *paired route-month* fields can do as well as receiver totals.

The paired-route arm observes 10 reporting bins on each of the 31 largest annual OD routes, selected only by the V310 annual route volume; the 11th bin follows from its annual route total. These 31 routes carry 77.56% of annual OD volume. The receiver arm observes 10 reporting bins for each of the 31 destinations, with the 11th implied. Both use 310 independent fields. A two-copy LP enforces identical measurements and maximizes the feasible target diameter over all possible measurement values.

The estimand is still imported source-weighted emissions divided by NBS above-size local generation, not consumption CIF or hourly risk. All data/support assumptions follow V321/V323, including residual EF support `[0,1] kgCO₂e/kWh`.

## Results

| Added field set | Fields | Mean diameter | Median diameter | Mean paired reduction vs no added fields | Median reduction |
|---|---:|---:|---:|---:|---:|
| None, five-category supply | 0 | 3.0384 | 1.7353 | — | — |
| Receiver monthly totals | 310 | 2.6584 | 1.5046 | 8.03% | 0.30% |
| Top-31 route monthly flows | 310 | 2.6613 | 1.4592 | 7.24% | 0.00% |
| Both sets | 620 | 2.6238 | 1.4592 | 9.11% | 0.30% |

The top-31 route design is positive in 154/341 province-period targets and in 14/31 province averages; receiver totals are positive in 187/341 cells and 17/31 province averages. The mean paired-reduction difference, route minus receiver, is −0.79 percentage points, with a province-cluster bootstrap 95% interval of −3.03 to +1.19 percentage points and only 6/31 provinces favoring routes. Thus the test does **not** resolve a meaningful advantage for either single 310-field design. Combining them gives a 1.08 percentage-point mean gain over receiver totals alone at twice the field budget; median benefit remains small.

Under total-generation-only supply caps, both measurement designs again have essentially zero value. Their value depends on the five-category source-side constraints already being present.

## Interpretation

Annual-flow-ranked sparse OD monitoring is a plausible alternative to full provincial receipt reporting: 31 routes cover over three quarters of annual route volume, and the worst-case average information value is statistically indistinguishable from 31 province receipt series under this proxy feasible set. It is not a clear breakthrough. Both designs have zero or negligible median cellwise gains and help a minority of provinces. The route heuristic was frozen from annual flow volume; it is not an optimized measurement design.

The strongest next method question is whether an *input-only* sparse-field selection rule—using annual OD, source-factor contrast, and supply caps but no target labels—can materially improve on both the top-volume route set and the receiver totals across years. This directly tests whether the contribution can be an adaptive, interpretable minimum-monitoring design rather than a generic flow-tracing LP. It must be compared on unseen years or regions and under wider residual-factor support before claiming a paper-level contribution.

## Limitations

- V310 OD and NBS-derived generation are from separate accounting sources and are not reconciled.
- The five-category source factors and `[0,1]` residual support are proxies; V322 documents support sensitivity.
- Monthly generation covers above-designated-size enterprises; the annual remainder is not allocated across months.
- The destination denominator is local reported generation. Demand, losses, and an independently validated consumption-side target are unavailable here.
- Measurement values are selected adversarially to compute worst-case conditional diameters. Results are not expected reductions for a particular observed dataset.
- Top-31 routes are one deterministic selection rule. A null or near-tie does not establish that all sparse OD designs are equivalent.

Input hashes, all 31 selected routes, field counts, and monotonicity audit are in `summary.json`. All measurement arms are verified not to widen bounds.

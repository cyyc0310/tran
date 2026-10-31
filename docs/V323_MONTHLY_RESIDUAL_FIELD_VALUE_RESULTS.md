# V323 results: minimax value of monthly residual-generation fields

**Run date:** 2026-10-30  
**Protocol:** [V323 protocol](V323_MONTHLY_RESIDUAL_FIELD_VALUE_PROTOCOL.md)  
**Reproduction:** `python scripts/v323_monthly_residual_field_value.py`  
**Machine-readable output:** `results/v323-monthly-residual-field-value-r1/summary.json` and `province_period_diameters.csv`

## Question and estimand

The public monthly NBS-derived table covers above-designated-size generation, whereas the same-source annual table covers all plants. V320/V321 therefore leave a known annual generation remainder whose monthly allocation is unobserved. This experiment asks whether releasing 310 province-period values for that remainder (31 provinces × 10 periods; the 11th is implied by the annual total) would reduce source-weighted direct-import exposure bounds more than 310 monthly province receipt totals.

For each province-period target, the LP maximizes the conditional diameter over every feasible pair of hidden systems consistent with the same hypothetical released field values. For the residual-profile arm, the monthly profile itself is shared between the two copies and is optimized subject to its known annual sum. This is a **worst-case-over-releases** criterion; it does not select a favorable or estimated monthly profile. The target is imported source-weighted emissions divided by local reported generation, not consumption CIF or hourly risk.

## Results

With five monthly generation categories and residual emission-factor support `[0, 1.0] kgCO₂e/kWh`:

| Added fields | Independent fields | Mean diameter | Median diameter | Mean paired reduction vs no added fields | Median paired reduction |
|---|---:|---:|---:|---:|---:|
| None | 0 | 3.0384 | 1.7353 | — | — |
| Monthly residual-generation profile | 310 | 3.0384 | 1.7353 | 0.00% | 0.00% |
| Monthly receipt totals | 310 | 2.6584 | 1.5046 | 8.03% | 0.30% |
| Both field sets | 620 | 2.6584 | 1.5046 | 8.03% | 0.30% |

The paired receipt-field reduction is positive in 187/341 province-period targets and in the province-average for 17/31 provinces. The province-cluster bootstrap 95% interval for the mean reduction is 4.05%–12.71%. In the equal-budget paired comparison, residual-profile fields are worse than receipt fields by 11.43 percentage points on average (95% interval −19.06 to −5.21 percentage points). Adding both sets gives no measurable gain over receipt fields alone in this worst-case criterion.

Under total-generation-only supply constraints, both hypothetical field sets have essentially zero value (receipt mean reduction 0.0008%; profile 0%). This is evidence of **complementarity**: in this feasible set, receipt totals help only after observed fuel-category caps restrict the available source mix. The exact field value is therefore conditional on the five-category supply information and the stated residual-factor support.

## Interpretation

The annual remainder’s monthly timing is not the main unidentified object for this endpoint. Even if its 10 monthly province allocations were released exactly, an adversary can still vary which destinations receive power and which source intensities are assigned within the feasible set, so the worst conditional diameter does not fall. Monthly receipt totals constrain the receiving side and do cut the average bound once source categories are known, but gains are highly uneven: the median reduction is only 0.30%, and 14 of 31 provinces do not show a positive province-average reduction.

For the next information-value step, prioritize a realistic receiver-side monthly field paired with the five-category generation report. The user has indicated public monthly reporting includes generation by all energy types; that can support the supply-side schema after the exact headings, units, and scope are verified. It does not substitute for receipt/consumption accounting. A useful next design should test *coarse destination × month receipts* or a small set of paired source-destination flows under the same field budget, rather than measuring the residual generation profile alone. No measured receipts are yet established in the audited public data used here, so this is a research prioritization, not an availability claim.

## Limitations and audit

- Monthly generation is above-designated-size coverage; annual remainder is annual all-plant generation minus that monthly coverage, not a directly observed sub-annual series.
- V310 annual OD routes come from an external accounting dataset and are not reconciled to the NBS-derived tables.
- The `[0,1]` residual factor support uses a declared proxy endpoint, not an engineering bound; V322 shows the field-value conclusions are support-sensitive and should be revisited at wider supports.
- The denominator is local reported generation. There is no monthly consumption balance, loss accounting, or validated recipient-side CIF label here.
- The 11 reporting bins are Jan–Feb and March–December. This is a structural partial-identification experiment, not a prediction-coverage test and not evidence of hourly risk calibration.
- These values are worst-case conditional diameters over hypothetical field releases. They do not estimate the expected benefit under a particular empirical distribution of monthly receipts.

Input and protocol hashes, missing-category handling, and cellwise monotonicity audit are recorded in `summary.json`. Both added-information constraint sets are verified not to widen any bound; the combined arm does not widen bounds versus either single field set.

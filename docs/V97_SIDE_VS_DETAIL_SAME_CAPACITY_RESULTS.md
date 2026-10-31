# V97: threshold-side monthly energy matches detailed margins under the same coarse capacities

**Result: the V96 candidate survives the matched-capacity comparison.** On the exact same 358 EIA region-months, V97 applies identical lagged thermal/nuclear/hydro/wind/solar capacity envelopes to (a) the threshold-side monthly statistic and (b) detailed monthly fuel margins. This removes the capacity-resolution confound from V96.

## Matched frontier

| τ | Target | Detailed month + coarse capacity: width | Threshold-side month + same capacity: width | Positive / negative / abstain, detailed → side |
|---:|---:|---:|---:|---:|
| 600 | ≥1% | 211 h | 214 h | 6.9 / 56.0 / 37.1% → 4.3 / 56.0 / 39.7% |
| 600 | ≥5% | 211 h | 214 h | 4.8 / 57.3 / 37.9% → 2.6 / 57.0 / 40.3% |
| 600 | ≥10% | 211 h | 214 h | 3.6 / 60.9 / 35.6% → 1.7 / 60.9 / 37.5% |
| 900 | ≥1% | 137 h | 142 h | 0 / 70.3 / 29.7% → 0 / 70.3 / 29.7% |
| 900 | ≥5% | 137 h | 142 h | 0 / 70.3 / 29.7% → 0 / 70.3 / 29.7% |
| 900 | ≥10% | 137 h | 142 h | 0 / 72.5 / 27.5% → 0 / 72.5 / 27.5% |

The threshold-side interval matches detailed monthly endpoints exactly on 2,088/2,148 rows; all 60 differences are one hour wider. With the same coarse capacity, 1,797/2,148 rows match endpoints exactly. The other 351 are wider by a mean 4.9 hours (maximum 84 hours); there are no rows where the threshold-side set is narrower than the detailed set. The average width penalty is only 3–5 hours against widths of 137–211 hours.

All observed K values are contained. Detailed monthly sets are nested in threshold-side sets, and detailed-month-plus-coarse-capacity sets are nested in threshold-side-plus-same-capacity sets. Side sets are also nested in the coarsest thermal-total budget. The assertions passed for all rows.

## Why this changes the research assessment

This is stronger than a comparison to fuel-specific capacity: at a fixed, coarse technology-capacity budget, one threshold-specific monthly energy aggregate almost reproduces the full detailed-fuel identification frontier. That suggests an interpretable and potentially useful algorithmic object: a risk-threshold-indexed sufficient-statistic map from fuel MWh to an exact K-set, with explicit EF-side uncertainty and capacity constraints. It can reduce required local data from all fuel margins to one risk-side sum for a specified threshold.

This is now a credible **paper candidate mechanism**, but not yet a verified breakthrough. It is on oracle-coarsened EIA records, fixed EF intervals, and a retrospective capacity-selected cohort. Next, stress ±1/2/5/10% error in the side-energy field, compare against simple scalar anchoring and standard optimization bounds, and seek a real independent hourly CIF reference. China source-field availability also remains unverified.

Protocol: [V97 protocol](V97_SIDE_VS_DETAIL_SAME_CAPACITY_PROTOCOL.md). Implementation: [V97 script](../scripts/v97_side_vs_detail_same_capacity.py). Outputs: [`results/v97-side-vs-detail-same-capacity-r1`](../results/v97-side-vs-detail-same-capacity-r1/).

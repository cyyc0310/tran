# V138 — Three-Day + Capacity Gains Survive Field Error, Positive Certificates Do Not

The frozen 358 region-month cohort produced 10,740 threshold/error/decision rows. All zero-error endpoints reproduce V136. Nominal proxy K containment, detailed-envelope containment and monotone expansion pass with zero failures; none of the 3,782 blocks needs a capacity fallback.

| Threshold | Zero-error width | ±10% risk-side field error | Monthly detailed + capacity | ≥1% positive certificates, zero → ±10% |
|---:|---:|---:|---:|---:|
| 600 | 191.15 h | 193.96 h | 222.14 h | 0.71% → 0% |
| 900 | 148.60 h | 150.24 h | 148.57 h | 0% → 0% |

At τ=600 the width advantage over monthly detailed fuel + capacity remains 12.7% at ±10% error. The ≥1% negative-certificate rate stays 58.04%, while abstention rises from 41.25% to 41.96%. At τ=900 the ≥1% negative-certificate rate remains 70.35%. At τ=600 and ≥5% tail frequency, the already rare positive certificates disappear at ±2% error.

These are sensitivity results with exact thermal/clean totals and exact lagged EIA capacity. They do not establish Chinese reporting accuracy. Three-day statistics also exceed the user's currently confirmed monthly information budget; this contract is an additional-information experiment, not the deployment input assumption. The user's subsequent clarification that all monthly generation categories are available makes detailed monthly margins the primary local-data contract.

Reproduction: [protocol](V138_THREE_DAY_CAPACITY_FIELD_ERROR_PROTOCOL.md), [runner](../scripts/v138_three_day_capacity_field_error.py), and `results/v138-three-day-capacity-field-error-r1/`. The first invocation stopped before data loading because it requested an absent V136 column; the corrected runner selects the saved 1% decision row for each region/month/threshold, with no change to the mathematical protocol.

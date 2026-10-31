# V93: coarse Chinese-style monthly fields still gain from annual capacity, but lose fuel-detail value at high thresholds

**Status: the information-budget effect survives coarsening; not a China validation or algorithm breakthrough.** V93 fixes the V90 cohort at 358 retrospectively capacity-feasible BA-months and hides EIA's monthly coal/gas/oil/other margins. The only monthly generation fields retained are thermal, nuclear, hydro, wind, and solar totals, matching the common public-report grouping. Thermal EF is allowed to vary adversarially between 400 and 1000 g/kWh in event and complement hours; clean group EF is zero, all under the same fixed proxy convention.

## Same-cohort frontier

| τ | Tail target | Coarse monthly fields only: positive / negative / abstain | Coarse monthly + annual group capacity: positive / negative / abstain | Mean K width: coarse month → + capacity | Detailed monthly + capacity width (V90) |
|---:|---:|---:|---:|---:|---:|
| 100 | ≥1% | 5.2 / 19.0 / 75.8% | 75.8 / 19.0 / 5.2% | 544 → 425 h | 419 h |
| 100 | ≥5% | 5.2 / 19.0 / 75.8% | 75.2 / 19.0 / 5.8% | 544 → 425 h | 419 h |
| 100 | ≥10% | 5.2 / 19.0 / 75.8% | 73.7 / 19.0 / 7.2% | 544 → 425 h | 419 h |
| 300 | ≥1% | 5.2 / 19.0 / 75.8% | 41.7 / 19.0 / 39.3% | 545 → 488 h | 442 h |
| 300 | ≥5% | 5.2 / 19.0 / 75.8% | 36.7 / 19.0 / 44.2% | 545 → 488 h | 442 h |
| 300 | ≥10% | 5.2 / 19.0 / 75.8% | 30.0 / 19.0 / 51.0% | 545 → 488 h | 442 h |

The capacity signal survives removal of detailed monthly fossil fuel composition. At τ=100, the information loss is small. At τ=300, coarse categories widen the mean interval by about 47 hours relative to detailed monthly fuel margins plus capacity and lower positive certification by about 10.1 percentage points for the ≥1% target and 3.3 points for ≥5%; the ≥10% positive rate is similar, with slightly more abstention. Coarse monthly categories alone provide essentially no decision gain over the detailed monthly-margin baseline in this cohort.

## Set checks

All 2,148 scored rows contain observed K. The capacity-informed set is nested within the coarse-month-only set. On every row, the coarse-budget set contains the detailed monthly-margin set, and the coarse-month-plus-capacity set contains V90's detailed-month-plus-capacity endpoint interval; no endpoint nesting failures occurred. Thus the frontier loss is attributable to hidden monthly fuel composition, while annual physical capacity still adds information. These are logical identification checks on oracle-coarsened EIA records, not calibrated forecast coverage.

## Relevance to China and innovation status

V93 more closely matches common Chinese monthly reports than V90 because it does not assume monthly coal/gas/oil decomposition. Yet its monthly categories are still reconstructed by aggregating EIA hourly fuel records, and its annual capacity is EIA-860, not a matched Chinese provincial source. Thermal EF range and capacity alignment require source-specific audit before applying these bounds.

The strongest candidate paper framing is now a **risk-specific information-budget benchmark**: detailed fuel-month fields vs thermal/clean monthly groups vs annual technology capacity, with selective tail-count identification and uncertainty retained. This is a substantive empirical/mechanism result, but it is not yet a new forecasting method and the EIA cohort is retrospectively selected. To claim algorithmic novelty, next derive robust bounds over uncertain within-thermal EF and capacity mapping, then evaluate the locked information contract on independent hourly CIF/region data.

Protocol: [V93 protocol](V93_CHINA_BUDGET_COARSENING_PROTOCOL.md). Script: [V93 implementation](../scripts/v93_china_budget_coarsening.py). Outputs: [`results/v93-china-budget-coarsening-r1`](../results/v93-china-budget-coarsening-r1/).

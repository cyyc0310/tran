# V96: a threshold-side monthly MWh summary nearly reproduces detailed fuel margins

**Candidate breakthrough: target-specific EF-side energy is nearly sufficient for the monthly tail-count set.** V96 tests a single risk-directed statistic on the same 358 retrospectively coarse-capacity-feasible EIA months. At τ=600 the statistic is monthly coal+oil generation (the proxy EF side above 600); at τ=900 it is coal generation. Within each side, fuel EF remains interval-valued. The evaluated targets are ≥1/5/10% monthly exceedance frequency.

## Main result

| τ | Monthly information | Mean K width without capacity | Positive / negative / abstain | Width with coarse annual technology capacity | Positive / negative / abstain with capacity at ≥1% |
|---:|---|---:|---:|---:|---:|
| 600 | Thermal/nuclear/hydro/wind/solar totals | 582 h | 0.0 / 19.0 / 81.0% | 555 h | 0.0 / 19.0 / 81.0% |
| 600 | Add coal+oil EF-side total | 316 h | 0.0 / 56.0 / 44.0% | 214 h | 4.3 / 56.0 / 39.7% |
| 600 | Detailed fuel monthly margins | 316 h | 0.0 / 56.0 / 44.0% | Detailed fuel capacity¹: 185 h | 5.1 / 55.4 / 39.5% |
| 900 | Thermal/nuclear/hydro/wind/solar totals | 582 h | 0.0 / 19.0 / 81.0% | 439 h | 0.0 / 19.0 / 81.0% |
| 900 | Add coal EF-side total | 213 h | 0.0 / 70.3 / 29.7% | 142 h | 0.0 / 70.3 / 29.7% |
| 900 | Detailed fuel monthly margins | 213 h | 0.0 / 70.3 / 29.7% | Detailed fuel capacity¹: 100 h | 0.0 / 70.6 / 29.4% |

¹Fuel-specific capacity is reported on the separately audited 326-month intersection; other coarse-budget rows use all 358 months.

Without capacity, threshold-side energy matches the detailed monthly-margin interval endpoints on 2,088/2,148 rows. On the remaining 60 rows, its interval is only one hour wider; it is never narrower than the detailed-margin set. Thus the individual monthly coal/gas/oil values appear largely unnecessary for these tail-count endpoints once their energy is aggregated by which side of τ their proxy EF occupies.

With capacity, the same risk-side statistic recovers most of the fine monthly signal while retaining coarse technology capacities: widths are 214 vs 185 h at τ=600 and 142 vs 100 h at τ=900. All 2,148 observed K values are contained; the detailed monthly-margin sets nest within the side-statistic sets; the side-statistic capacity sets nest within the coarser thermal-total sets; and the fuel-capacity feasible sets are contained by the side-statistic capacity sets.

## Why this is a stronger candidate

The information requirement is tied to the user’s risk threshold. At τ=900, one coal-month MWh field captures the only thermal source above threshold. At τ=600, the high side is coal+oil, so coal alone is insufficient but a single coal+oil aggregate nearly reproduces the detailed-fuel margin frontier. This suggests a general design: publish or estimate monthly generation on the EF-above-threshold side, rather than requiring every fuel split or inventing a universal hourly shape.

This is a candidate information-sufficiency mechanism, not yet the final breakthrough: results are oracle-aggregated from EIA hourly data, use proxy EF intervals, and use a retrospective capacity-screened cohort. The coal+oil threshold-side field may not be a directly published Chinese statistic. V97 should compare this summary against detailed monthly margins plus the **same coarse technology capacity** on identical rows and test measurement error in the threshold-side aggregate. Then validate the frozen field contract on independent system data and actual CIF if possible.

Protocol: [V96 protocol](V96_THRESHOLD_SIDE_ENERGY_PROTOCOL.md). Implementation: [V96 script](../scripts/v96_threshold_side_energy.py). Outputs: [`results/v96-threshold-side-energy-r1`](../results/v96-threshold-side-energy-r1/).

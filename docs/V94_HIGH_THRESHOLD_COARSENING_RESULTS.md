# V94: high-threshold CIF tails need local thermal composition information

**Finding: common coarse monthly categories plus annual capacity cannot certify high-threshold positive tail risk in this EIA proxy cohort.** V94 evaluates τ=600 and 900 g/kWh, where the assumed thermal EF support [400,1000] crosses the threshold. The 358-region-month cohort is fixed from V90. Detailed fuel capacity is reported on its own 326-month feasible intersection; the coarse-budget result retains all 358 months.

## Coarse-budget results on all 358 months

| τ | Tail target | Detailed monthly fuel margins: positive / negative / abstain | Coarse month categories only | Coarse month + annual technology capacity | Mean K width: detailed month → coarse month → coarse + capacity |
|---:|---:|---:|---:|---:|---:|
| 600 | ≥1% | 0.0 / 56.0 / 44.0% | 0.0 / 19.0 / 81.0% | 0.0 / 19.0 / 81.0% | 316 → 582 → 555 h |
| 600 | ≥5% | 0.0 / 56.0 / 44.0% | 0.0 / 19.0 / 81.0% | 0.0 / 19.0 / 81.0% | 316 → 582 → 555 h |
| 600 | ≥10% | 0.0 / 56.0 / 44.0% | 0.0 / 19.0 / 81.0% | 0.0 / 19.0 / 81.0% | 316 → 582 → 555 h |
| 900 | ≥1% | 0.0 / 70.3 / 29.7% | 0.0 / 19.0 / 81.0% | 0.0 / 19.0 / 81.0% | 213 → 582 → 439 h |
| 900 | ≥5% | 0.0 / 70.3 / 29.7% | 0.0 / 19.0 / 81.0% | 0.0 / 19.0 / 81.0% | 213 → 582 → 439 h |
| 900 | ≥10% | 0.0 / 70.3 / 29.7% | 0.0 / 19.0 / 81.0% | 0.0 / 22.4 / 77.6% | 213 → 582 → 439 h |

At both high thresholds, the coarse thermal total leaves the positive-tail rate at zero across the whole cohort, while abstention rises to 81%. Annual coarse capacity reduces interval width but does not produce a positive certificate. By contrast, detailed monthly fuel margins can rule out the target on 56.0% of rows at τ=600 and 70.3% at τ=900. This is consistent with a threshold-specific minimum-information result: coal/gas/oil composition is much less consequential when all thermal proxy factors lie above the threshold, but becomes central when the threshold cuts through the thermal factor range.

## Fuel-capacity paired comparison

Fuel-specific capacity passes its peak-and-energy audit on 326/589 quality months; V94's fixed-cohort intersection contains 326 months. On this subset, fuel-specific capacity (detailed monthly margins plus annual fuel caps) has mean widths of 185 h at τ=600 and 100 h at τ=900. For τ=600 it gives a 5.1% positive certification rate across ≥1/5/10% targets; at τ=900 it gives no positive certificates, but negative certification rises to 70.6–74.6%. These figures come from the smaller paired subset and should not be compared directly with the 358-month coarse percentages without the same-subset table in `fine_capacity_paired_frontier.csv`.

All 2,148 coarse rows contain observed K. The detailed monthly-margin set is contained in the coarse-month set, the coarse-capacity set is nested within the coarse-month set, and on feasible fine-capacity months the coarse-capacity set contains the detailed-fuel-capacity set. No endpoint nesting failures occurred.

## Scientific implication

This is the strongest evidence so far that information requirements depend on the risk threshold. If China's public table has only total thermal generation, its annual technology capacity may help constrain broad tail counts but cannot support positive high-threshold certificates under the tested proxy EF uncertainty. A local monthly coal share, fuel-weighted thermal EF, or another source of thermal composition may be the minimum additional statistic for τ values that cross the thermal EF range. The experiment does not prove the exact minimum statistic, because it tests coarse categories against full fuel margins rather than a one-field-at-a-time coal/gas ablation.

V94 remains a retrospective EIA coarsening experiment. Monthly categories are oracle-aggregated from hourly records; the capacity-feasible cohort was selected using target-hour audits; fixed EFs are proxies, not actual CIF. It supports a risk-specific information-budget benchmark and a falsifiable minimum-data question, not a China forecast or a novel forecasting algorithm.

Protocol: [V94 protocol](V94_HIGH_THRESHOLD_COARSENING_PROTOCOL.md). Implementation: [V94 script](../scripts/v94_high_threshold_coarsening.py). Outputs: [`results/v94-high-threshold-coarsening-r2`](../results/v94-high-threshold-coarsening-r2/).

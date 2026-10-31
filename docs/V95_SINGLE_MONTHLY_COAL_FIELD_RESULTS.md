# V95: one monthly coal field is sufficient at τ=900 but not τ=600

**Result: the minimum useful local field depends on the risk threshold.** V95 adds only monthly coal MWh to the coarse monthly thermal/nuclear/hydro/wind/solar table. Residual noncoal thermal EF is unknown in [400,800], while capacity is shared at the thermal-group level. The main frontier uses the 358-month V90 cohort; detailed fuel-capacity comparison uses the stricter 326-month audited intersection.

## All 358 months: adding coal MWh

| τ | Information set | Mean K width | Positive / negative / abstain at ≥1% | At ≥5% | At ≥10% |
|---:|---|---:|---:|---:|---:|
| 600 | Coarse monthly fields | 582 h | 0.0 / 19.0 / 81.0% | same | same |
| 600 | + coal monthly MWh | 582 h | 0.0 / 19.0 / 81.0% | same | same |
| 600 | Coarse monthly + annual group capacity | 555 h | 0.0 / 19.0 / 81.0% | same | same |
| 600 | + coal monthly MWh + annual group capacity | 514 h | 6.9 / 19.0 / 74.0% | 4.8 / 19.0 / 76.2% | 3.6 / 19.0 / 77.4% |
| 900 | Coarse monthly fields | 582 h | 0.0 / 19.0 / 81.0% | same | same |
| 900 | + coal monthly MWh | 213 h | 0.0 / 70.3 / 29.7% | same | same | same |
| 900 | Coarse monthly + annual group capacity | 439 h | 0.0 / 19.0 / 81.0% | same | 0.0 / 22.4 / 77.6% |
| 900 | + coal monthly MWh + annual group capacity | 142 h | 0.0 / 70.3 / 29.7% | same | same |

At τ=900, coal is the only thermal proxy category above threshold. Monthly coal energy therefore restores nearly all the detailed-month no-capacity identification: width 213 h vs 213 h, and negative certification 70.3% vs 70.3%. Adding coarse annual capacity further narrows the coal-field set to 142 h. At τ=600, both coal and oil exceed threshold while gas/other do not; coal alone barely changes the no-capacity set (582→582 h) and leaves most cases abstained even with annual capacity.

On the 326-month fuel-capacity paired subset, fine capacity narrows more (184.6 h at τ=600 and 100.3 h at τ=900). The coal-field capacity set is wider but includes the fine set endpoints on every paired row; its regional macro widths are 514 h and 142 h. This shows the coal field is highly informative for τ=900 but does not replace all subfuel detail for τ=600.

## Scientific implication

This is a concrete minimum-local-information hypothesis: rather than needing every thermal subfuel margin for every risk, the minimum statistic may be the monthly MWh on the EF side above a chosen threshold. At τ=900 that side is coal; at τ=600 it includes coal and oil, so a coal-only field is insufficient. A next direct test is a threshold-specific monthly high-EF-side aggregate (e.g., coal+oil for τ=600; coal for τ=900) with the rest left interval-valued. If this matches detailed-margin frontiers, the method contribution could be a target-aware information contract plus a sharp partial-identification algorithm.

This remains an oracle EIA coarsening on a retrospectively screened cohort, using fixed proxy EFs and no China source file or actual hourly CIF. It is a promising, falsifiable mechanism finding, not a validated China predictor.

Protocol: [V95 protocol](V95_SINGLE_MONTHLY_COAL_FIELD_PROTOCOL.md). Implementation: [V95 script](../scripts/v95_single_monthly_coal_field.py). Outputs: [`results/v95-single-monthly-coal-field-r2`](../results/v95-single-monthly-coal-field-r2/).

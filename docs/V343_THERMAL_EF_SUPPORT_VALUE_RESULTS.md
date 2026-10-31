# V343 value of thermal-factor information at τ=300

## Experiment

Using the exact V342 solver, we evaluated 330 complete Chinese province-periods at thermal direct-EF values 400, 500, 600, 700, 800, 900 and 1,000 g/kWh. Other technology direct operational factors remain zero. The comparison keeps the NBS five-category monthly generation margins fixed and contrasts monthly-only, 2022 lagged CEC capacity, and retrospective 2023 capacity. The support bands are sensitivity scenarios, not observed uncertainty intervals.

## Support-width frontier

| Arm | Thermal EF support | Mean τ=300 K-hull width | Reduction from [400,1000] |
| --- | --- | ---: | ---: |
| Monthly categories only | 400–1000 | 795.8 h | — |
| Monthly categories only | 600–800 | 795.4 h | 0.4 h |
| Monthly categories only | exactly 700 | 795.4 h | 0.4 h |
| + 2022 lagged capacity | 400–1000 | 682.0 h | — |
| + 2022 lagged capacity | 500–900 | 586.1 h | 96.0 h |
| + 2022 lagged capacity | 600–800 | 532.1 h | 149.9 h |
| + 2022 lagged capacity | exactly 700 | 499.9 h | 182.2 h |
| + 2023 capacity | 400–1000 | 686.2 h | — |
| + 2023 capacity | 500–900 | 594.2 h | 92.0 h |
| + 2023 capacity | 600–800 | 542.6 h | 143.6 h |
| + 2023 capacity | exactly 700 | 511.7 h | 174.5 h |

## Interpretation

This gives a useful “minimum additional information” result. Knowing the thermal factor exactly barely narrows the τ=300 hull from monthly category totals alone (0.4 hours), because the hourly allocation remains unconstrained. Combined with installed-capacity limits, narrowing the thermal-factor support from 400–1,000 to exactly 700 reduces the mean hull by 182.2 hours. Most of that reduction is a complementarity between factor information and capacity: a factor scalar has little value without a physical output envelope.

Even after granting the oracle exact thermal EF=700 and 2022 lagged capacity, the average τ=300 hull remains about 500 hours in a month. The dominant remaining uncertainty is hourly generation timing/dispatch and within-technology composition. The current public five-category data do not contain that information. Thus the data support a conditional partial-identification framework and a targeted value-of-information result, but still do not validate a credible hourly point/interval prediction for an unseen province.

The EF value 700 and support bands were selected as sensitivity points, not calibrated local Chinese thermal factors. Historical capacity release timing and hour-level validity are also unverified. The result should guide the next data collection experiment: measure whether a legally available province-level thermal-fuel or heat-rate split can narrow the factor support enough to matter, while recognizing that a separate temporal-shape signal is still needed for a narrow τ=300 risk interval.

## Reproducibility

- Protocol: [V343 protocol](V343_THERMAL_EF_SUPPORT_VALUE_PROTOCOL.md)
- Runner: [V343 script](../scripts/v343_thermal_ef_support_value.py)
- Outputs: `results/v343-thermal-ef-support-value-r1/`

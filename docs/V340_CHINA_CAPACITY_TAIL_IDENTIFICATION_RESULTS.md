# V340 corrected China capacity tail identification results

## V339 correction

The first V339 outputs are invalid for comparing the 2022 and 2023 CEC capacity vintages. Its loop applied the current vintage to both named capacity arms and emitted duplicate province-period rows, so the reported arms appeared identical by construction. V339's files remain preserved; use the corrected V340 output below.

## Corrected result

V340 contains 2,046 distinct province-period × threshold × tail-rate rows (31 provinces × 11 NBS periods × 2 thresholds × 3 risk rates). It uses the 2022 capacity table for the lagged arm and the 2023 table for the retrospective arm. The monthly-only hull width is 795.4 hours at τ=100 and 795.8 hours at τ=300.

| Threshold | Information arm | Mean K-hull width | Narrowing vs monthly only | Positive certification | Abstention |
| --- | --- | ---: | ---: | ---: | ---: |
| 100 g/kWh | Monthly + 2022 lagged capacity | 468.9 h | 41.1% | 87.1–89.1% | 10.9–12.9% |
| 100 g/kWh | Monthly + 2023 capacity (retrospective) | 482.0 h | 39.4% | 87.1–89.1% | 10.9–12.9% |
| 300 g/kWh | Monthly + 2022 lagged capacity | 675.9 h | 15.1% | 38.1–46.9% | 53.1–61.9% |
| 300 g/kWh | Monthly + 2023 capacity (retrospective) | 680.0 h | 14.6% | 37.8–46.9% | 53.1–62.2% |

Across 2,046 rows, lagged and same-year capacity hull endpoints match on 534 rows; lagged capacity is narrower on 1,440 rows, while same-year capacity is narrower on 72. This supports a vintage-sensitive information-value result under the declared assumptions, not a forecast result. The positive/abstain frequencies are conditional model certificates, not empirical coverage or calibrated probabilities.

## Scientific limits and next gate

The input consists of public monthly NBS technology totals and manually transcribed CEC yearbook capacity. Eleven Tianjin periods omit a hydro category; that amount is kept in an unresolved residual group. The absolute difference between the sum of reported categories and the published total has median 4.16 GWh and maximum 23.25 GWh per province-period, so the target pool is defined by reported category margins and discrepancies are audited rather than silently balanced. The reported monthly production is below both 2022 and 2023 nameplate-capacity energy envelopes for every reported thermal/hydro/wind/solar group, but this only checks monthly-average feasibility.

The emissions target is a generation-side direct-CIF scenario, not measured hourly CIF and not a consumption-side MEE factor. Fixed nonthermal direct factor zero, thermal `[400,1000]` g/kWh, and unresolved `[0,1000]` are modeling supports. Province-wide total capacity is a loose upper bound for nuclear and unresolved output. Hourly label truth and capacity validity at every hour are unavailable. Therefore the width reduction may justify a carefully scoped partial-identification benchmark contribution, but does not yet establish a predictive breakthrough. The next methodological gate is to calculate the sharp K-set under the same group-capacity contract and compare it to the current outer hull; if the width benefit disappears or the outer set is materially loose, the claimed method contribution weakens.

## Reproducibility

- Protocol: [V340 correction protocol](V340_CHINA_CAPACITY_TAIL_IDENTIFICATION_CORRECTION.md)
- Script: [corrected V340 runner](../scripts/v340_china_capacity_tail_identification_corrected.py)
- Results: `results/v340-china-capacity-tail-identification-corrected-r1/`

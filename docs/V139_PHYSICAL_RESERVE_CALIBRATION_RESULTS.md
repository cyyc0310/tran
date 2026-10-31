# V139 — Capacity-Contract Correction and Physical-Reserve Calibration

## Input audit changed the interpretation of V120

V120 r2 called 589 region-months capacity-feasible. Joining its saved rows to V100's input-only monthly-energy audit shows that 76 months (12.9%) violate the capacity contract. Of these, 43 τ=600 rows and one τ=900 row return a purported physical interval that excludes the robust proxy K endpoint. All 1,026 threshold endpoint pairs in the remaining 513 months exactly match V103's audited results.

The cause is the V92 solver's endpoint shortcuts: they checked monthly emissions but bypassed global monthly energy versus capacity. An infeasible system could therefore return `(0,)` or `(T,)`. The solver now returns `None` before evaluating any K when a group monthly margin exceeds T times its capacity. Four saved boundary examples verify both infeasible endpoints and their feasible counterparts.

The V120 performance table must not be used as overall method evidence. Target-side capacity bounds in studies with a separate energy check are not automatically invalidated, but source support thresholds derived from V120 need correction. See [correction scope](CAPACITY_CONTRACT_CORRECTION.md).

## Corrected paired comparison

V139 retains all 589 months and all 25 regions. The 76 capacity-unavailable months receive [0,T] as a conservative outer hull; their availability decision uses monthly energy and capacity, not target K. Every method uses this same budget, and each outer region is excluded from all calibration scores. All 17,670 scored rows pass denominator, containment, interval and nesting audits.

At alpha=0.10:

| Threshold | Method | Region-equal monthly coverage | Mean width | All-month region coverage |
|---:|---|---:|---:|---:|
| 600 | Corrected physical hull / availability fallback | 100% | 297.63 h | 100% |
| 600 | Absolute anchor calibration + physical clipping | 99.47% | 201.99 h | 92% |
| 600 | Physical-width normalized calibration | 99.47% | 180.05 h | 92% |
| 900 | Corrected physical hull / availability fallback | 100% | 225.33 h | 100% |
| 900 | Absolute anchor calibration + physical clipping | 97.17% | 33.60 h | 92% |
| 900 | Physical-width normalized calibration | 97.17% | 27.30 h | 92% |

Relative to the absolute anchor, normalization reduces width by 10.9% / 18.7% with unchanged empirical coverage at alpha=0.10. But at τ=900, coverage among region-months with K_high>0 is only 78.96%; the high overall coverage conceals rare-tail failures in JEA and SC. At alpha=0.05 this conditional coverage is 92.17%, monthly coverage is 98.5%, and mean width is 116.71 h. At alpha=0.20, overall τ=900 coverage remains 95.15% while conditional positive-tail coverage collapses to 4.35%. The complete alpha frontier is saved; no operating point is selected for confirmation.

The directional-reserve and physical-width methods produce identical endpoints on every row. The monthly-constant anchor is at a physical endpoint or spans the full factor-box hull, so the directional formula adds no distinct mechanism in this experiment. They must not be counted as two innovations.

## What this supports

Physical normalization is a useful baseline improvement under the corrected budget. It does not yet protect rare systems with different hourly coal/clean co-occurrence. Normalized conformal scores and interval-valued conformal inference are established methods; scaling alone is insufficient novelty. The more useful next question is whether complete monthly composition and utilization can identify the rare-tail regime before observing hourly labels, compared with strong conditional quantile models under the same data budget.

This remains an opened U.S. source-pool development experiment with assumed EF bounds and target-month totals. Chinese table headers, publication times, compatible capacity and true hourly average-CIF labels still require validation.

Reproduction: [protocol](V139_PHYSICAL_RESERVE_CALIBRATION_PROTOCOL.md), [runner](../scripts/v139_physical_reserve_calibration.py), `results/v139-physical-reserve-calibration-r1/`. The input audit and protocol amendment preceded scoring; the original preparation attempt stopped on the containment precondition.

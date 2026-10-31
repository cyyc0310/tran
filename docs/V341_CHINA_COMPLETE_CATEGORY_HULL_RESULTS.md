# V341 complete-category China generation-risk hull

## Data contract

The user supplied the NBS monthly-data query. The matching provincial monthly series used in this experiment is described by OREO as NBS-compiled generation in five groups—total, thermal, hydro, nuclear, solar and wind—in GWh ([dataset page](https://oreo-data.org/dataset/cn-prov-monthly-generation-by-tech)). NBS monthly releases identify their generation figures as output from above-designated-size industrial enterprises; the scope is not all generation ([NBS release and scope note](https://www.stats.gov.cn/sj/zxfb/202601/t20260119_1962329.html)). The five groups are technology categories. Thermal combines coal, oil, gas, waste heat/pressure/gas and biomass generation, so these monthly data do not provide a fuel-specific thermal mix ([NBS definition of thermal power](https://www.stats.gov.cn/sj/pcsj/jjpc/5jp/zk/html/zb02.pdf)).

## Correction and cohort

V339 was invalid for capacity-vintage comparison because its 2022 and 2023 arms were mislabeled inside the vintage loop and province-period rows were duplicated. V340 repaired those issues but allowed the 11 Tianjin periods with missing hydro to enter after setting the missing category to zero. V341 excludes these incomplete periods; it does not use the published total to impute hydro because category sums also differ from the reported total. All 341 province-periods are retained in the audit, and 330 complete periods across 30 provinces enter the frontier.

Rows: 1,980 unique complete province-period × threshold × rate combinations. Arms: monthly categories alone; monthly categories with lagged 2022 CEC capacity; and monthly categories with 2023 CEC capacity as a retrospective reference. Capacity is tested as a conservative technology-group upper bound, with total provincial capacity as a loose nuclear cap.

## Frontier

| Threshold | Arm | Mean outer-hull width | Reduction vs monthly-only | Positive certificate | Abstention |
| --- | --- | ---: | ---: | ---: | ---: |
| 100 g/kWh | Monthly only | 795.4 h | — | 0% | 100% |
| 100 g/kWh | + 2022 lagged capacity | 469.9 h | 40.9% | 86.7–88.8% | 11.2–13.3% |
| 100 g/kWh | + 2023 capacity | 483.4 h | 39.2% | 86.7–88.8% | 11.2–13.3% |
| 300 g/kWh | Monthly only | 795.8 h | — | 0% | 100% |
| 300 g/kWh | + 2022 lagged capacity | 682.0 h | 14.3% | 36.1–45.2% | 54.8–63.9% |
| 300 g/kWh | + 2023 capacity | 686.2 h | 13.8% | 35.8–45.2% | 54.8–64.2% |

These are conditional mathematical certificates, not empirical coverage, probabilities, or forecast scores. The emitted object is an outer hull `[min K,max K]`, not the full feasible integer set. Capacity narrows the hull more for τ=100 than τ=300, while many τ=300 cases still abstain.

## Interpretation

This is a useful information-budget result: technology-level monthly output plus coarse installed capacity can constrain some hour-tail counts in a large Chinese province panel, even though the five-category monthly series alone leaves almost the full month unresolved. It is not yet a method breakthrough for credible prediction. Results use a fixed-factor direct-generation scenario (thermal factor support 400–1000 g/kWh; direct nonthermal factor zero), and the target has no Chinese hourly generation/CIF labels. CEC vintage and hourly coverage alignment are retrospective and not independently validated. The data only cover above-designated-size industrial output, and no fuel-level breakdown exists within thermal generation.

The next decisive method experiment is a sharp solver for the five-group K set under the same constraints, checked against the current outer hull. If the outer-hull widths are materially conservative, the contribution can move toward an exact, assumption-transparent partial-identification method. If they are already tight, novelty must come from a validated public-information benchmark and source/availability design, not relabeling this ordinary constraint as a new predictor.

## Reproducibility

- Protocol: [V341 protocol](V341_CHINA_COMPLETE_CATEGORY_HULL_PROTOCOL.md)
- Runner: [V341 script](../scripts/v341_china_complete_category_hull.py)
- Final guarded output: `results/v341-china-complete-category-hull-r2/` (r1 is retained as the pre-guard run)

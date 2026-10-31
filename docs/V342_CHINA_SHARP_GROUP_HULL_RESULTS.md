# V342 sharp group-capacity hull audit

## Result

An exact fixed-factor group-capacity solver was applied to the same 330 complete province-periods and 2022/2023 capacity arms as V341. It checks monthly energy-capacity feasibility before evaluating any K endpoint and requires positive generation in both event and non-event blocks. It stores the complete feasible integer K set at each thermal-factor support endpoint, 400 and 1,000 g/kWh.

| τ (g/kWh) | Arm | Mean exact hull width | Mean V341 outer-hull width | Rows with equal endpoints | Rows whose endpoint-set union fills every integer in hull |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | Monthly categories | 795.42 h | 795.42 h | 330/330 | 330/330 |
| 100 | + 2022 lagged capacity | 469.86 h | 469.86 h | 330/330 | 330/330 |
| 100 | + 2023 capacity | 483.36 h | 483.36 h | 330/330 | 330/330 |
| 300 | Monthly categories | 795.77 h | 795.77 h | 330/330 | 330/330 |
| 300 | + 2022 lagged capacity | 682.05 h | 682.05 h | 330/330 | 330/330 |
| 300 | + 2023 capacity | 686.20 h | 686.20 h | 330/330 | 330/330 |

Across all 660 province-period × threshold combinations in each arm, the independent sharp solver matched the V341 outer hull's lower and upper endpoints exactly. For every row, the union of the complete K sets at thermal EF 400 and 1,000 contained every integer between those endpoints; it had zero holes. For any fixed hourly allocation, increasing the thermal EF cannot lower K, so the full K set over continuous support `[400,1000]` cannot extend outside these extreme hull endpoints. Thus the interval hull is sharp, and its integer range is fully populated, under this fixed-group model and the audited capacity contract.

## What this establishes—and what it does not

This is a strong information-bound result for the Chinese public-data contract: monthly technology-group margins alone leave almost the entire month feasible; annual coarse capacity sharply reduces that set at τ=100, but much less at τ=300. The bound is not merely a loose outer approximation under the stated mathematical model.

It is not a new exact-set algorithm: the project already contains the analogous sharp group-capacity construction in V92/V102, and optimization-based identification from aggregates is established. The incremental value here is the China-specific five-technology information budget and the measurement of when public annual capacity does or does not resolve a higher-threshold tail. It still lacks Chinese hourly generation/CIF labels, historical release-vintage validation, and verification that province capacity fully covers the NBS above-designated-size generation pool at every hour. The direct thermal EF is treated as a month-constant unknown within `[400,1000]`; actual fuel mix and unit-level heat rates vary.

Paper positioning should therefore remain a **partial-identification benchmark under coarse public information**, not a transferable hourly CIF prediction model. A credible next test is a same-budget sensitivity study of the minimum additional public information needed to shrink τ=300 bounds, particularly the thermal group's effective factor support or a finer legal thermal-fuel split. Claims about predictive reliability require an independent hourly label source.

## Reproducibility

- Protocol: [V342 protocol](V342_CHINA_SHARP_GROUP_HULL_PROTOCOL.md)
- Runner: [V342 script](../scripts/v342_china_sharp_group_hull.py)
- Final guarded outputs: `results/v342-china-sharp-group-hull-r4/` (r1–r3 are retained as intermediate code/data-integrity iterations)

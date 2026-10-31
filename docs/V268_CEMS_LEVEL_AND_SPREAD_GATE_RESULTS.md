# V268 results: factor level shift changes which target months fail the gate

Date: 2026-10-30. Cross-system stress test using measured U.S. CEMS fuel factors; not China validation.

## Matched 16 BA-month comparison

The same quarter-specific state-factor dispersion radii from V267 were applied under two center choices. One centers the interval on project nominal factors `[1000, 800, 400] g/kWh`; the other centers each interval on the measured CEMS national fuel-quarter factor. Both are applied to the same V240 EIA monthly fuel margins and the same 500 g/kWh threshold.

| Factor uncertainty contract | Threshold-crossing BA-months | Mean lower monthly intensity | Mean upper monthly intensity |
|---|---:|---:|---:|
| Project nominal level + CEMS spread | 4/16 | 332.1 | 402.6 |
| CEMS level + CEMS spread | 2/16 | 321.4 | 389.1 |
| Project nominal + uniform ±10% | 4/16 | — | — |

The project-centered CEMS spread crosses AZPS October and PACE April/July/October. The CEMS-centered interval crosses PACE July/October. The uniform ±10% reference crosses AZPS October and PACE April/July/October. Equal aggregate counts can hide differences in *which* targets require local factor information.

The measured CEMS factor centers themselves also differ from the project proxy centers: approximate coal 914–924 vs 1000, gas 408–419 vs 400, and oil 681–748 vs 800 g/kWh. This is a mix of system/fuel definitions and should not be treated as a direct correction to the EIA proxy or a China transfer estimate.

## Research consequence

The useful refinement is that “EF precision” is not one scalar radius. A threshold-side gate depends jointly on (a) fuel-specific factor **level** and (b) fuel-specific **uncertainty width**. A common ±r box can predict the wrong cases for which temporal data are useless. This is a stronger, testable method hypothesis: estimate the target's coarse factor level first, retain asymmetric fuel-specific factor sets, then choose whether additional temporal information can reduce the tail-risk identified set.

This is not yet a breakthrough claim. CEMS state-quarter heterogeneity is U.S.-specific, not a target-region set; the small oil sample is weak; all evaluated BA-months were opened; and the result is a deterministic identification gate, not a coverage-tested forecast. Next validation should use directly measured target fuel factors and must preserve unseen regions.

## Correction to initial run

The initial r1 target means divided emissions by fossil generation only, not all-source generation. The CEMS centers and radii were correct; the target gate means and crossing results were not. Corrected r2 uses total generation including zero-direct-EF sources and supersedes r1.

Reproducibility: `scripts/v268_cems_level_and_spread_gate.py`, `docs/V268_CEMS_LEVEL_AND_SPREAD_GATE_PROTOCOL.md`, and corrected `results/v268-cems-level-and-spread-gate-r2/`.


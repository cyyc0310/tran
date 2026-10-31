# V112 results: worst-region calibrated factor-robust anchor

## Outcome

Replacing V111's pooled radius with the maximum source-region q90 yields radius 0.4649 at τ=600 (set by LDWP) and 0.3564 at τ=900 (set by SC). The resulting anchor interval contains the low-to-high factor-box K range in all 60 target region-month/threshold rows.

| Threshold | Worst-region anchor width | Monthly-only robust set width | Capacity robust set width | Factor-box coverage |
|---|---:|---:|---:|---:|
| 600 gCO₂/kWh | 335.5 h | 722.2 h | 434.1 h | 100% |
| 900 gCO₂/kWh | 257.4 h | 577.6 h | 270.0 h | 100% |

The robust anchor intervals issue no positive or negative certificates at 1%, 5%, or 10% tail targets: the worst-region radius leaves them too wide to classify. In contrast, the capacity set produces negative certificates for 20% of target months at τ=900 for 1%/5% tail rates, and 23.3% at 10%. At τ=600 both approaches abstain throughout.

## Interpretation

This is a coverage-efficiency versus decision-rate tradeoff. The worst-region anchor is narrower at both thresholds and has 100% empirical factor-box coverage on this small heldout cohort, while the physical set method is slightly narrower at τ=900 and provides a few valid negative certificates. Neither method certifies positive high-tail risk; most decisions remain abstentions. The anchor's empirical source-region q90 is not a distribution-free guarantee, and five regions/six months cannot establish new-domain coverage.

Current evidence favors a benchmark that reports the complete frontier rather than a single winning method. It also weakens the case for claiming a point-prediction algorithm: the simplest local mean anchor is adequate in this low-tail cohort, while conservative methods trade efficiency for structural guarantees. Genuine breakthrough evidence would require a future period or Chinese hourly labels with nontrivial tail events, where methods differ on hard cases.

## Reproducibility

- Frozen protocol: `docs/V112_WORST_REGION_ROBUST_ANCHOR_PROTOCOL.md`
- Runner: `scripts/v112_worst_region_robust_anchor.py`
- Outputs: `results/v112-worst-region-robust-anchor-r1/`
- Source hashes and the worst-radius region are recorded in `summary.json` and `source_region_q90.csv`.

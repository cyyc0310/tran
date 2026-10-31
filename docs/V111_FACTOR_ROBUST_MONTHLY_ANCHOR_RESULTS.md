# V111 results: factor-box monthly anchor calibration

## Outcome

The same ±10% thermal factor box is applied to the monthly-mean anchor and the identified-set methods. Calibration uses 589 source region-months from 25 regions in 2024–2025; targets are the fixed 30 V104 region-months. Each calibration error is the worst residual across low/nominal/high factor vectors. The pooled region-balanced empirical q90 radii are 0.0639 tail fraction at τ=600 and 0 at τ=900.

| Threshold | Robust anchor interval width | Capacity identified-set width | Anchor covers factor-box K range |
|---|---:|---:|---:|
| 600 gCO₂/kWh | 83.9 h | 434.1 h | 28/30 months (93.3%) |
| 900 gCO₂/kWh | 0 h | 270.0 h | 30/30 months (100%) |

The two τ=600 misses are FPC 2026-01 and 2026-02. The anchor interval upper bounds are 84 and 78 hours, while the factor-box proxy K upper endpoints are 119 and 131 hours. The capacity set covers them but is much wider. At τ=900 the factor-box proxy K range is K=0 for every target month; the zero-width anchor result is therefore a degenerate easy-case result, not evidence of general uncertainty calibration.

## Interpretation

The pooled calibrated anchor is much sharper than partial identification but fails the desired factor-box coverage in one region. This is exactly the failure mode that pooling can hide. A worst-source-region radius is needed as a stronger baseline. V111 does not establish a breakthrough in either direction: it shows why an empirically sharp point/interval baseline and robust partial identification occupy different points on a coverage-efficiency frontier.

## Reproducibility

- Frozen protocol: `docs/V111_FACTOR_ROBUST_MONTHLY_ANCHOR_PROTOCOL.md`
- Runner: `scripts/v111_factor_robust_monthly_anchor.py`
- Outputs: `results/v111-factor-robust-monthly-anchor-r1/`
- Source and code hashes are in `summary.json`.

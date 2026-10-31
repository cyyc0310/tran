# V106 results: coarse public-style monthly data on unseen regions

## Outcome

The frozen five-region V104 cohort contributes all 30 Jan–Jun 2026 months. Every month passes the label-coverage and lagged group-energy feasibility rules. We withheld coal/gas/oil/other monthly splits, retained thermal/nuclear/hydro/wind/solar group totals, and allowed thermal EF to vary over [400,1000] gCO₂/kWh independently in event and complement hours. The derived sets contain observed fixed-factor proxy K in all rows and are nested when annual group capacity is added.

## Information-budget frontier

| Threshold | Coarse monthly only width | With lagged group capacity | Reduction | Negative certificate rate, capacity condition | Abstention at 10% tail target |
|---|---:|---:|---:|---:|---:|
| 600 gCO₂/kWh | 722.6 h | 607.2 h | 115.4 h (16.0%) | 0% | 100% |
| 900 gCO₂/kWh | 722.2 h | 410.8 h | 311.5 h (43.1%) | 0% at 1%/5%; 6.7% at 10% | 93.3% |

No positive certificates occur. Under this coarse data contract, only 6.7% of region-month decisions can rule out a ≥10% tail at τ=900. Detailed monthly fuel margins plus the same capacity and a narrower ±10% factor box yielded 23.3% negative certificates on the same threshold/rate in V105, with a mean robust width of 270.0 h versus 410.8 h here. This contrasts two distinct information and factor-uncertainty regimes, not a controlled same-factor comparison.

## Interpretation

Annual coarse technology capacity adds partial-identification information even when internal thermal shares are hidden, but it does not resolve the hourly temporal ordering: most high-threshold decisions remain abstentions. The gap to fuel-resolved margins indicates that coal-versus-gas/oil monthly composition is a high-value local field near τ=900. The experiment does not establish that all Chinese statistical releases provide a comparable split or capacity vintage; the field contract and release time still need direct source validation. It also uses target-month EIA monthly totals and a U.S. capacity analogue, so it is retrospective mechanism evidence rather than a China forecast.

## Reproducibility

- Frozen protocol: `docs/V106_UNSEEN_CHINA_STYLE_COARSE_INFO_PROTOCOL.md`
- Runner: `scripts/v106_unseen_china_style_coarse_info.py`
- Outputs: `results/v106-unseen-china-style-coarse-info-r1/`
- Source, protocol, and script hashes are in `summary.json`.

# V135 — Shared Factor Uncertainty Does Not Further Tighten the Three-Day Set

## Results

The 358-region-month V129 cohort produced 2,148 scored rows and zero containment failures. For every threshold, tail rate, region-month, the coherent all-low/all-high factor endpoint hull matched V132's blockwise factor-support interval in both endpoints and decisions.

| Threshold | Monthly risk-side width | Three-day blockwise width | Three-day coherent-factor width | Three-day detailed oracle |
|---:|---:|---:|---:|---:|
| τ=600 | 316.08 h | 246.10 h | 246.10 h | 245.10 h |
| τ=900 | 213.50 h | 197.95 h | 197.95 h | 196.77 h |

The all-low and all-high scenario sets jointly contain nominal K and the same-bin detailed-fuel robust envelope, and they sit inside V132's wider-support reference.

## Interpretation

Enforcing cross-block factor coherence does not produce additional shrinkage here. The three-day interval width is therefore not being driven by independent factor relaxation across days; the remaining uncertainty comes from how hourly generation can be arranged within each reporting block under the released aggregates. This closes one plausible optimization path and points the next experiment toward operational constraints or within-block shape summaries.

This negative finding is limited to the V129 EIA cohort, its broad factor-support assumptions, and three-day bins. The factors are stress intervals rather than validated local distributions; no Chinese hourly labels, publication error, or capacity information are included.

## Reproduction

- Protocol: [V135 protocol](V135_COHERENT_FACTOR_MULTIDAY_PROTOCOL.md)
- Script: [V135 runner](../scripts/v135_coherent_factor_multiday.py)
- Outputs: `results/v135-coherent-factor-multiday-r1/`
- Protocol SHA-256: `acc1bddd8f871baf414fbf3359fb6a89050176dd5c2d39d64237ad08bafc3144`
- Script SHA-256: `4d8c96e93a1dbc1e7f138eabf09200a45a94ee3aa97635506d2f0fea3c055def`
- Summary SHA-256: `b6462568713304416158ec95036c76c434cff0fd6430f8dab3ca5995266d03d7`

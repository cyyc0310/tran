# V134 — Three-Day Contract Transfers in Width, Not in Tail Classification

## Results

The frozen V132 three-day rule was applied to all 15 V113 2026Q3 region-months for CPLE, FPC, IPCO, PNM, and SOCO. Every month had complete hourly coverage, all 90 scored rows passed, and the reconstructed nominal proxy K matched V113's existing labels. Robust side intervals contained both nominal K and the detailed-fuel oracle envelope.

| Threshold | Monthly risk-side width | Three-day risk-side width | Width decrease | Three-day detailed oracle | ≥1% negative / positive / abstain |
|---:|---:|---:|---:|---:|---:|
| τ=600 | 735.0 h | 599.6 h | 135.4 h (18.4%) | 599.2 h | 0% / 0% / 100% |
| τ=900 | 588.0 h | 561.3 h | 26.7 h (4.5%) | 561.3 h | 20% / 0% / 80% |

The three-day summary nearly reaches the same-bin detailed-fuel oracle: its mean width is only 0.4 h wider at τ=600 and equal at τ=900. Yet the intervals remain too wide to establish positive high-tail risk. At the ≥10% target, τ=600 produces 13.3% negative certificates; at τ=900 the decision rates remain unchanged from the monthly contract.

## Interpretation

The transfer probe reproduces the direction of V132's information gain outside its 21-region development cohort, but the gain is much smaller in relative terms at τ=900 and does not improve positive-tail detection. This strengthens a data-budget/partial-identification result: once temporal bins are supplied, daily fuel detail adds little. It does not establish a useful high-risk forecast.

The 2026Q3 outcomes were already opened for earlier TransCIF methods. This is a frozen-rule method-transfer probe, not blind confirmation. The cohort has only five regions and three months, uses U.S. EIA fixed-factor proxy labels and assumed factor intervals, excludes capacity, and contains no actual Chinese hourly CIF labels. Future time or genuinely new region data must be held out before confirmatory claims.

## Reproduction

- Protocol: [V134 protocol](V134_FROZEN_THREE_DAY_Q3_TRANSFER_PROTOCOL.md)
- Script: [V134 runner](../scripts/v134_frozen_three_day_q3_transfer.py)
- Outputs: `results/v134-frozen-three-day-q3-transfer-r1/`
- Protocol SHA-256: `478268d0b4f3d78efcd989d208b5d6b77ed77ccff3b11fcc63f2e59c56f47497`
- Script SHA-256: `d0245893f9fb2bbbabe51df96d4dae740378fc65ebfd6babd8e7ac2199617dad`
- Summary SHA-256: `de39e7aa52c7bfbc3c82cf21dd68d3913b46ed8cdf1f5589aff7b2c9fe1b6a77`

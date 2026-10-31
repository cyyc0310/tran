# V136 — Coarse Capacity Makes the Three-Day Contract More Efficient at τ=600

## Results

On the V129 frozen 358 region-month cohort, V136 scored 2,148 rows across two thresholds and three tail-rate targets. There were no nominal-K or detailed-envelope failures. All 3,782 three-day blocks passed the lagged EIA group-capacity energy feasibility check; the fallback-to-no-capacity rate was 0%.

Equal-region mean widths (hours) are:

| Threshold | Monthly detailed fuel + capacity | Monthly risk-side + capacity | Three-day risk-side, no capacity | Three-day risk-side + capacity | Three-day detailed oracle + capacity |
|---:|---:|---:|---:|---:|---:|
| τ=600 | 222.14 | 232.55 | 246.10 | **191.15** | 181.93 |
| τ=900 | 148.57 | 162.36 | 197.95 | 148.60 | 135.37 |

At τ=600, the three-day risk-side+capacity interval is 14.0% narrower than the strong monthly detailed-fuel+capacity baseline and only 9.23 h wider than the same-bin detailed-fuel oracle. Its ≥1% decision is 0.71% positive, 58.04% negative, and 41.25% abstaining, versus 4.52% / 56.01% / 39.47% for the monthly detailed baseline. It trades away some rare positive certificates while issuing more negative certificates and producing a narrower interval. At τ=900, the width is effectively tied with the monthly detailed baseline (148.60 vs 148.57 h); decision rates change little. The 14% width reduction at τ=600 is not universal: it is narrower in 9 of 21 regions, and equal or wider in the others.

## Interpretation

This is the strongest incremental result in the current information-contract branch: threshold-specific three-day MWh summaries plus coarse lagged capacity can beat monthly detailed fuel composition on τ=600 interval width while retaining full proxy-set containment. The gain is threshold-dependent and the high-threshold result is a tie. The selective-risk tradeoff means the method does not dominate the strong baseline on every decision metric.

This remains a retrospective EIA oracle-coarsening experiment. The 0% capacity fallback rate only shows compatibility in this selected U.S. cohort; it does not establish Chinese provincial capacity availability, vintage alignment, category boundaries, or legal publication. The factors are sensitivity intervals, and the hourly labels are fixed-factor proxies. The next step is to freeze the combined rule and apply it to the 2026Q3 region-time cohort as a method-transfer probe, while clearly marking that cohort's outcomes were previously opened for other TransCIF methods.

## Reproduction

- Protocol: [V136 protocol](V136_THREE_DAY_CAPACITY_COMBINATION_PROTOCOL.md)
- Script: [V136 runner](../scripts/v136_three_day_capacity_combination.py)
- Outputs: `results/v136-three-day-capacity-combination-r1/`
- Protocol SHA-256: `eb413edd2a8a6c41efe8fc38358624d224d445fec82f45a205e82787c419a3fc`
- Script SHA-256: `3a848ac0f4a658e4bed8b7d49376a0ff76905cb356d904bad1321d18bd228ca2`
- Summary SHA-256: `d9b0084ff1c065689bc59eb0e9539f642a65299c0b5e1f62f485fef2440a6a79`

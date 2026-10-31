# V137 — Frozen Three-Day + Capacity Rule Transfers to 2026Q3, with Small Width Gains

## Results

The V136 three-day risk-side+capacity rule was applied unchanged to the 15 V113 Q3 region-months. All hours matched V113, the Q3 nominal K values matched the saved V113 rows, 160 three-day blocks passed capacity feasibility with no fallback, and all 90 scoring rows passed containment checks.

| Threshold | Monthly detailed + capacity width | Three-day risk-side + capacity width | Three-day detailed oracle width | ≥1% negative / positive / abstain |
|---:|---:|---:|---:|---:|
| τ=600 | 455.47 h | 423.20 h | 400.07 h | 0% / 0% / 100% |
| τ=900 | 277.60 h | 260.93 h | 257.80 h | 20% / 0% / 80% |

Relative to monthly detailed fuel plus capacity, the three-day rule narrows mean width by 7.1% at τ=600 and 6.0% at τ=900. It remains 23.13 h and 3.13 h wider than the same-bin detailed-fuel oracle. At ≥1% tail risk, the certificate/abstention rates do not improve; at τ=600 and ≥10%, it certifies no exceedance in 13.3% of region-months, with the rest abstaining.

## Interpretation

The frozen rule retains a smaller width advantage in the Q3 cohort than in the 2024–25 development cohort. The transfer probe supports the direction of the information-efficiency result, not a useful positive-tail forecast. At τ=600/900 most Q3 cases still abstain because the outer intervals remain broad.

This is not blind confirmation: all five regions and Q3 outcomes had previously been opened for other TransCIF methods. It is also a small five-region, three-month EIA proxy study with assumed factor bounds and no Chinese hourly CIF labels. The zero capacity fallback rate only describes these U.S. EIA regions; it does not validate provincial Chinese capacity records.

## Reproduction

- Protocol: [V137 protocol](V137_FROZEN_THREE_DAY_CAPACITY_Q3_PROTOCOL.md)
- Script: [V137 runner](../scripts/v137_frozen_three_day_capacity_q3.py)
- Outputs: `results/v137-frozen-three-day-capacity-q3-r1/`
- Protocol SHA-256: `5b2f29edf95fcdce2c97e45123f0e8824fa5ffd823ce3cf63359ee4d3454a8d3`
- Script SHA-256: `517d7b53d987c5d6b58660020346c7e2d89c695c8f0c0c832d4243591cd3dbd1`
- Summary SHA-256: `c136fb933f10e5ab686724dfd307dbd016ada0768dced8745730dfb856a46f51`

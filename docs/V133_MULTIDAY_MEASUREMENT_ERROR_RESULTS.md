# V133 — Three-Day Risk-Side Width Is Stable to the Tested Field Error; Rare Certificates Are Not

## Results

The frozen 358 region-month cohort yielded 32,220 rows across two thresholds, three target rates, three bin sizes, and five measurement-error bounds. Eight synthetic closed-form checks matched V130's general measurement-error solver. The full run had zero nominal-containment, detailed-envelope, zero-error equality, or monotone-expansion failures.

At the three-day reporting resolution and ≥1% target:

| Threshold | Risk-side MWh error | Mean interval width | Positive / negative / abstain |
|---:|---:|---:|---:|
| τ=600 | 0% | 246.100 h | 0.48% / 56.01% / 43.52% |
| τ=600 | 1% | 246.111 h | 0.48% / 56.01% / 43.52% |
| τ=600 | 2% | 246.128 h | 0.48% / 56.01% / 43.52% |
| τ=600 | 5% | 246.178 h | 0% / 56.01% / 43.99% |
| τ=600 | 10% | 246.217 h | 0% / 56.01% / 43.99% |
| τ=900 | 0% | 197.954 h | 0% / 70.35% / 29.65% |
| τ=900 | 10% | 197.963 h | 0% / 70.35% / 29.65% |

The interval width changes by only 0.12 h at τ=600 and 0.01 h at τ=900 as the risk-side amount error expands to ±10%. The rare τ=600 positive certificate disappears between ±2% and ±5%. The coarse bounds are often already determined by strict endpoint feasibility, so a moderate uncertainty band changes few integer endpoints; this does not make the certificate itself robust.

## Interpretation

The candidate three-day information contract retains its width advantage under this bounded-risk-field error stress. Its tiny positive decision gain does not: with ≥5% error, τ=600 positive certificates disappear and abstention returns to the monthly-contract rate. A defensible claim should therefore emphasize interval efficiency and explicit abstention, not positive-tail warning skill.

This is an adversarial sensitivity exercise, not a measured data-quality study. Only the risk-side MWh field is perturbed; thermal total and clean generation remain exact. Errors are applied independently by block, and bounds are summed conservatively. The result is still based on retrospective EIA coarsening, fixed-factor proxy labels, and the V90-selected cohort—not Chinese publication uncertainty or actual hourly CIF labels.

## Reproduction

- Protocol: [V133 protocol](V133_MULTIDAY_MEASUREMENT_ERROR_PROTOCOL.md)
- Script: [V133 runner](../scripts/v133_multiday_measurement_error.py)
- Outputs: `results/v133-multiday-measurement-error-r1/`
- Protocol SHA-256: `b48b0a9985b896177f15381787b6a54d804f22ece003686dcb4ab95da570d73e`
- Script SHA-256: `38354060e5d7952b3eeea05ef2e6c619cb7246f235da416c5e178bac115039b6`
- Summary SHA-256: `2e4e225f124909f799d3ec1ce4e24b850275c790e98de08bdbd1befd218e52d8`

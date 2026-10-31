# V50 nested source validation for fossil-role strength

Run date: 2026-10-27. Protocol: [V50_NESTED_FOSSIL_ROLE_PROTOCOL.md](V50_NESTED_FOSSIL_ROLE_PROTOCOL.md). The outer held source was excluded from both the inner λ selection and the outer coefficient fit. The run used 70 outer prediction packages and ten external label packages; it opened no target/unseen source and refit no point or risk model.

## Nested performance

The inner selector chose λ=.25 in 33 of 70 outer×held-source cases, λ=0 in 22, λ=.50 in 13, λ=.75 in 2, and λ=1 in none. This concentration on conservative increments supports shrinkage, while the variation shows the strength is source/context dependent.

| Model | Hourly CRPS | Daily level MAE | 90% coverage | Width | Interval score | Clipping |
|---|---:|---:|---:|---:|---:|---:|
| Raw anchor | 125.14 | 144.61 | 88.50% | 527.98 | 712.16 | 5.66% |
| EF-only monthly mix | 71.43 | 93.47 | 95.27% | 478.64 | 510.42 | 10.05% |
| Full fossil increment | 64.53 | 72.38 | 86.34% | 483.74 | 628.55 | 2.06% |
| Nested fossil increment | 66.62 | 86.74 | 94.87% | 481.31 | 517.98 | 3.88% |

Nested selection improves substantially on the raw anchor, and its lowest source-macro coverage is 85.48% (SEC; SCL 85.98%). Against the EF-only model, it improves CRPS by about 6.7%, with similar coverage and lower clipping. However, interval score increases from 510.42 to 517.98 and width from 478.64 to 481.31. Thus the fossil-role increment does not yet improve interval efficiency over the already stronger EF-only arm, despite improving proper point/distribution score.

## Decision

Nested shrinkage fixes the most obvious post-hoc λ selection problem and yields a credible source-held-out signal. It still falls short of the paper gate: the interval efficiency comparison is negative against EF-only, worst-source coverage is close to the floor, and all ten sources are prior development sources. This is not enough to freeze a breakthrough method or open a genuinely unseen region. Further work should target why corrected centers do not translate to sharper calibrated hourly intervals, and compare against same-budget strong risk models before confirmation.

## Artifacts

- Script: `scripts/probe_v50_nested_fossil_role.py`
- Output: `results/disjoint-calibration-v50-nested-fossil-role/`
- `summary.json` binds the protocol, code, frozen V50 seals/month inputs, metrics, and nested-selection table.

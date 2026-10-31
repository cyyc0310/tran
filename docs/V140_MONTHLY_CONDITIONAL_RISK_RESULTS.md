# V140 — Complete Monthly Composition Does Not Yet Make the Learned Variant Beat the Anchor

## Result under the confirmed monthly-only budget

All 589 region-months and 25 regions are retained. Two deterministic source splits give disjoint 12-region fit and 12-region calibration pools for every outer held-out region. Six methods, two budgets, two thresholds and two alpha levels produce 56,544 rows. Denominator, physical-containment, role separation, interval and alpha-nesting checks report zero failures. No new target region or future period was opened.

At alpha=0.10, entries show seed 41 / seed 83; both repetitions remain visible.

| Threshold | Monthly-only method | Macro coverage | Mean width | Coverage where K_high>0 |
|---:|---|---:|---:|---:|
| 600 | Absolute monthly anchor | 99.30% / 99.14% | 198.10 / 198.89 h | 97.62% / 97.18% |
| 600 | Direct CQR, all fit months | 92.95% / 96.00% | 176.04 / 190.14 h | 79.48% / 88.76% |
| 600 | Direct CQR, active fit months | 98.00% / 99.83% | 233.76 / 244.18 h | 94.40% / 99.54% |
| 600 | Physical-coordinate CQR | 97.67% / 99.33% | 235.65 / 243.37 h | 93.48% / 98.15% |
| 900 | Absolute monthly anchor | 98.00% / 98.50% | 50.13 / 86.83 h | 77.51% / 92.17% |
| 900 | Direct CQR, all fit months | 97.33% / 97.67% | 62.53 / 116.28 h | 79.83% / 87.83% |
| 900 | Direct CQR, active fit months | 97.33% / 97.67% | 64.37 / 129.68 h | 79.83% / 87.83% |
| 900 | Physical-coordinate CQR | 97.33% / 97.67% | 64.29 / 129.68 h | 79.83% / 87.83% |

The learned physical-coordinate variant does not improve the joint coverage/efficiency frontier over the simple anchor in both repetitions. Removing physically trivial months helps direct CQR's coverage, but its intervals become much wider. Under monthly-only inputs the physical coordinate is nearly the ordinary tail fraction, so there is little new geometry to learn.

## Optional capacity budget

At τ=600, physical-coordinate CQR has widths 197.85 / 181.23 h and coverage 95.50% / 97.83%. Normalized anchor calibration is both narrower (174.81 / 175.10 h) and more covered (99.12% / 99.14%). At τ=900 the coordinate model is also wider and less covered than the normalized anchor. A same-budget strong direct CQR control therefore does not rescue the proposed coordinate mechanism as a paper contribution. The full alpha=0.10/0.20 frontier and fixed-penalty robust interval scores are saved; no operating point is promoted to confirmation.

Capacity is still an optional U.S. research proxy. The monthly-only comparison is the relevant one for the user's currently confirmed Chinese data access.

## Source support is a concrete next bottleneck

A saved posthoc input-support audit shows that SC is the only source region with mean coal share above 60% (65.89%). Its 20/24 monthly coal shares exceed the maximum coal share of all other 24 regions. NWMT, the next most coal-heavy source, averages 48.17%; its maximum EF-box upper-tail fraction at τ=900 is only 0.287%, versus SC's 84.95%.

When SC is held out, the capacity-coordinate model's τ=600 monthly coverage is 0% / 45.83% across the two source splits; at τ=900 it is 41.67% in both. This establishes a visible source-support defect and a concentration of failure. It does not prove that adding coal-heavy donors will solve the hidden dispatch shift. That causal comparison is the next experiment: add historical donors from already examined regions, preserve the same held-out targets and calibration roles, and measure whether the increment survives simple anchored and conditional baselines. Richer source support must not be presented as an algorithmic innovation by itself.

Current evidence does not justify opening genuinely untouched confirmation regions. The original hourly CIF forecasting gate also remains unmet: these are monthly tail-count reconstruction experiments with target-month totals and assumed EF-box proxy labels, not hourly coverage/CRPS results or actual Chinese CIF validation.

## Reproduction and literature

- [Frozen protocol](V140_MONTHLY_CONDITIONAL_RISK_PROTOCOL.md), [runner](../scripts/v140_monthly_conditional_risk.py), output `results/v140-monthly-conditional-risk-r1/`.
- [Source-support audit](../scripts/v140_source_support_audit.py), output `results/v140-source-support-audit-r1/`.
- [Primary-source novelty boundary](V140_LITERATURE_AND_NOVELTY_BOUNDARY.md): CQR, interval-valued conformal inference, covariate-shift conformal inference and task-specific inverse-problem uncertainty already exist. A physical normalization or a newly named calibration layer is insufficient originality.

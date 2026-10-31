# V142 — Expanded Sources Do Not Resolve LGEE; Strong Anchors Remain Necessary

## Design and audit

PACE's 18 and LGEE's 24 historical months are evaluated as already examined development targets. Every target's identity is removed from fitting and calibration. Each seed uses 30 fit and 12 calibration source regions; fixed-function all-source/local anchors may use all 42 allowed source identities. The seven methods, two thresholds, two seeds and two alpha levels yield 2,352 rows. Role, denominator, physical-subset and alpha-nesting checks report zero failures.

## Alpha=0.10 results

Entries show seed 41 / seed 83. Source role variation is retained.

| Target / threshold | Method | Coverage | Mean width |
|---|---|---:|---:|
| LGEE / 600 | All-source anchor | 100% / 100% | 381.17 / 381.17 h |
| LGEE / 600 | Nearest-12-source anchor | 100% / 100% | 404.42 / 404.42 h |
| LGEE / 600 | Direct CQR, all fit months | 100% / 100% | 720.00 / 720.00 h |
| LGEE / 900 | All-source anchor | 95.83% / 95.83% | 691.63 / 691.63 h |
| LGEE / 900 | Nearest-12-source anchor | 100% / 100% | 707.38 / 707.38 h |
| LGEE / 900 | Direct CQR, all fit months | 12.50% / 45.83% | 612.71 / 668.92 h |
| LGEE / 900 | Direct CQR, active fit months | 12.50% / 62.50% | 612.71 / 687.13 h |
| PACE / 600 | All-source anchor | 94.44% / 94.44% | 535.00 / 535.00 h |
| PACE / 600 | Nearest-12-source anchor | 100% / 100% | 552.61 / 552.61 h |
| PACE / 600 | Direct CQR, all fit months | 100% / 100% | 730.11 / 617.78 h |
| PACE / 900 | All-source anchor | 94.44% / 94.44% | 14.94 / 14.94 h |
| PACE / 900 | Nearest-12-source anchor | 100% / 100% | 398.78 / 398.78 h |
| PACE / 900 | Direct CQR, all fit months | 94.44% / 100% | 17.50 / 69.39 h |

The source-support gain seen on the original 25-region pool does not generalize uniformly to these difficult identities. A local anchor has 100% empirical coverage in every displayed cell, but is often broad. CQR sometimes improves PACE efficiency, yet fails badly on LGEE at τ=900 and returns essentially the full physical monthly range at τ=600. Physical-coordinate CQR does not establish a distinct advantage; all methods and alpha levels remain in the saved frontier.

The seed-83 disjoint anchor at PACE/900 is particularly brittle: 77.78% endpoint coverage, 2.94 h width, and an 11.11% robust certificate-error rate at the ≥1% tail decision. Its calibration pool lacks adequate rare-tail support. The full-source anchor is more stable but still misses one PACE month. No target-informed adjustment is applied.

## Why LGEE needs two separate diagnoses

Its saved τ=900 factor-box response has K_low=0 and mean K_high=651.42 h. Any interval required to cover all factors in the assumed box must therefore be at least 651.42 h wide on average, even with the hourly generation allocation observed. The nearest-source anchor is only 55.96 h above this oracle floor. The learned model's smaller width often makes endpoint containment impossible.

At τ=600 the same factor-box endpoints are equal to the observed-hour count for every LGEE month. The factor-contract floor is zero there; a 720 h predicted width is therefore dispatch-transfer/model uncertainty, rather than factor ambiguity. This is the more substantial reducible-uncertainty target. [V143](V143_FACTOR_FLOOR_DIAGNOSTIC_RESULTS.md) saves this accounting over the full source panel and all predictions.

These floors are conditional on the project's assumed EF support and observed proxy allocation. They are not measured Chinese uncertainty bounds, nor are their oracle values available as target features.

## Decision

Do not promote source expansion plus CQR to a general method breakthrough. Keep strong all-source/local anchors, and prioritize a mechanism that transfers the composition-to-tail response without treating the factor-driven width as a calibration defect. Ordinary fuel recomposition already exists in this project and must be a same-budget baseline; simply renaming it or adding a conformal wrapper does not establish originality.

This remains retrospective monthly-tail development. No genuinely unopened confirmation region was accessed and no hourly forecasting/CRPS gate was passed.

Reproduction: [protocol](V142_PACE_LGEE_MONTHLY_HOLDOUT_PROTOCOL.md), [runner](../scripts/v142_pace_lgee_monthly_holdout.py), `results/v142-pace-lgee-monthly-holdout-r1/`.

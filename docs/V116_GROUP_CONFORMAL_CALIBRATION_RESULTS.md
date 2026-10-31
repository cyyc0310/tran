# V116 — Region-Grouped Split-Conformal Calibration Diagnostic Results

## Result

The region-level calibration score is the maximum monthly worst-case absolute anchor error per source region. With 25 regions and 10% nominal miscoverage, the frozen rank is 24/25. The resulting radii are 0.5282 at τ=600 and 0.1097 at τ=900. The V112 reference radii were 0.4649 and 0.3564, respectively.

Reapplying those radii to the already opened V115 spatial holdout gives 22/22 threshold rows contained by the factor-box proxy endpoint interval for the anchor, monthly-only set, capacity set, and anchor-capacity intersection. On 11 eligible region-months, region-equal mean widths are:

| Threshold | Anchor | Monthly-only | Capacity | Intersection | Intersection vs capacity |
|---|---:|---:|---:|---:|---:|
| τ=600 | 384.9 h | 183.8 h | 161.0 h | **97.2 h** | −39.6% |
| τ=900 | 79.6 h | 183.8 h | 126.3 h | **20.1 h** | −84.1% |

The τ=900 intersection is 69.3% narrower than the V115 intersection (65.5 h), largely because the new grouped radius is much smaller at that threshold. At τ=600 the more conservative grouped radius widens the V115 intersection from 85.3 h to 97.2 h. At all three tail rates, the negative/abstain/positive frontier is unchanged at 75%/25%/0%; the narrower sets create no new decisions. There are no positive certificates.

## What this supports

This is a useful calibration-design result: calibrating a maximum score per region avoids treating correlated months as independent and produces a finite-sample split-conformal radius under exchangeability of calibration and target regions. It is a promising ingredient for a future method combining cross-region calibration with a physically sharp feasible set. It also reveals threshold-specific behavior: the conservative grouped score helps at τ=600 but is substantially more efficient than V112 at τ=900.

## What this does not support

V115 target labels were already inspected before the V116 recalibration was proposed, so this is post-hoc exploratory evidence, not an independent confirmation. The target roster was selected by historical data completeness, and region exchangeability after that selection is not established. Split-conformal coverage is marginal, not conditional for any one region. The 100% result is empirical containment of a fixed-EF proxy's factor-box endpoints in only four regions and one quarter; it is not validation against Chinese hourly CIF, nor a month-ahead forecast result. The 75% negative certificates mostly arise in regions with no support for high-EF generation; the hard TEPC months remain abstentions.

## Next gate

Freeze the region-grouped score, rank, factor box, physical constraints, thresholds, selection rule, and evaluation code. The next cohort must be new in both region labels and time period, selected without target labels, and include months with nonzero high-threshold proxy events. Report coverage, width, decision frontier, and performance separately for zero-support and nonzero-support months. A real China CIF claim still requires same-boundary hourly generation and emissions labels.

## Reproduction

- Frozen protocol: `V116_GROUP_CONFORMAL_CALIBRATION_PROTOCOL.md`.
- Runner: `../scripts/v116_group_conformal_calibration_diagnostic.py`.
- Outputs and input hashes: `../results/v116-group-conformal-calibration-diagnostic-r1/`.

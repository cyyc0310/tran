# Frozen single-region confirmation: TVA 2026H1

Registered 2026-10-27, after selecting the development candidate on the ten V50 external regions and before opening TVA 2026H1 generation values or scores. This is a one-region, out-of-development-source confirmation of transfer behavior; it is not a multi-region guarantee and does not establish China transfer.

## Why TVA is eligible

Use the V6 preparation protocol's predeclared candidate order and metadata-only quality screen. TVA was the seventh qualifying candidate after the first six were selected by the fixed rule; its 2026H1 quality screen records 181/181 complete UTC days. The independent inventory at `results/coarse-dynamics-v9-verification/holdout-inventory.md` records no prior TVA score/prediction artifact and states that prior examination was limited to quality metadata. The V6 audit and raw-file identities are frozen inputs. TVA is not swapped if the experiment fails. The preselected target is exactly TVA 2026-01-01 through 2026-06-30.

## Frozen candidate and comparators

No further parameter, arm, lag, interval, source, feature, or method selection is allowed.

1. **Primary candidate:** frozen V50 `B0-SizeET-AlignedKeep` point paths and `MCStd` initial risk bounds; fit the EF-only daily location correction on all ten V50 external calibration regions using the registered source-equal weighted least-squares procedure; cross-fit that correction within the ten calibration regions (each calibration region's correction is trained on the other nine); calibrate separate equal-source 95th percentiles of lower and upper nonnegative excess scores; fit the final TVA correction using all ten sources and apply the frozen two tail quantiles. Use 90-day V50 support for all source calibration and target daily support fixed by the V6 quality screen. Preserve [0,1000] bounds and original clipping rules.
2. **Matched strong baseline:** same B0 paths, EF-only location correction, and MCStd base scores, with the existing equal-source symmetric 90% conformal quantile.
3. **Strong risk baselines at the same monthly-information budget:** same B0 paths and EF-only correction with HETScale and CQRplus, each using the existing equal-source symmetric 90% conformal score. Their correction and calibration construction is identical to the matched baseline; only the frozen risk head changes.
4. **Simple anchor:** uncorrected B0 point distribution, expanded around its daily mean by the fixed factor 2 and clipped to [0,1000], as used in the prior V50 comparison.

All source risk heads and point functions are the already-frozen V50 models. No target fit, target label calibration, threshold adjustment, or target-derived feature is allowed. For each of the seven V50 outer point functions, produce one TVA prediction; summarize the seven repeated views within each UTC day before reporting target metrics. These seven functions are not seven independent target regions.

## Information and temporal rules

- Target inputs may use only complete TVA monthly fuel-composition rows with `available_at <= origin`, `period_end <= origin`, and the fixed V6 month-end-plus-30-day release rule. Use the V6 native snapshot and its recorded schema/mapping. The 2026H1 TVA hourly labels are not an input, feature, donor, fit, or calibration source.
- The feature/model horizon is 2026H1 only. Use the preselected 181-day quality mask from the V6 audit; do not change it based on CIF scores. A discrepancy between the sealed target label support and the frozen 181-day mask invalidates the run; do not silently drop or replace target days.
- Since `disjoint_point_v50` and `risk_baselines_v50` guard their original source/date roles, implement the TVA role as a separate, auditable extension layer. Do not change frozen V50 modules, source artifacts, or their hashes. The extension may accept only region `TVA`, dates 2026H1, B0, and the specified risk heads; it must refuse any other target or role.
- The V6 source uses revised historical snapshots and an assumed 30-day release delay, not archived original publication vintages. Report this as a controlled-information experiment, not real-time issued-public validation. The target is an EIA generation-side fixed-EF proxy, not complete observed CIF.

## Required sealing sequence

1. Verify V50 freeze, prediction and calibration seals, all 70 source prediction packages, ten external labels, V6 audit and the nine raw native snapshot hashes. Bind these hashes and the complete code/configuration closure in a fresh output directory.
2. Generate and seal all TVA 2026H1 raw prediction paths, risk bounds, legal monthly feature rows, corrections, all baseline intervals, and exact scored-day keys. The prediction seal must state `TVA_hourly_label_values_opened=false` and include a complete seven-outer × 181-day grid. Verify no use of TVA labels in model fitting, correction fitting, or calibration.
3. Only after the prediction seal passes, construct TVA hourly proxy labels using the already-frozen V6 quality/mapping rules, verify the exact 181-day key set, then score the four methods. Seal scores and a receipt. Do not revise methods or thresholds after scoring.

## Outcomes and interpretation

Primary metrics: hourly 90% coverage, interval width, interval score (`width + 20 × lower excess + 20 × upper excess`), lower/upper miss rates, hourly CRPS when a valid 128-path predictive distribution is available, and clipping fraction. Report results by month and aggregate over the single TVA region. Compare the primary candidate to all three baselines listed above. Do not use hours, days, seven outer functions, or months as independent geographic replicates; no p-value or confidence claim about new-region performance is made.

The candidate's development evidence is treated as a preselected hypothesis. A favorable TVA result is one external-region success, not proof of universal or China transfer; a failure remains a valid negative confirmation and cannot trigger tuning or substitution. A nominal 90% interval is not called successful solely because coverage is high: interval score and width must improve over the matched and strong same-budget baselines while coverage remains at least 90%. The simple anchor is reported regardless of outcome. Algorithmic novelty remains unclaimed because asymmetric conformal and quantile-adaptive intervals are established methods.

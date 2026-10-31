# V50 coarse-mix conditional interval source-held-out test

Registered 2026-10-27 after the monthly full-mix risk association probe and a focused literature check. The test treats coarse-mix-conditioned score prediction as a baseline to evaluate, not a novelty claim. Localized/conditional conformal methods already exist.

## Objective

Test whether the full monthly EF-weighted fuel-mix feature's source-level p90 signal translates into actual held-source interval coverage and efficiency, relative to a single global threshold and simple/strong baselines. This tests empirical transfer among ten already-used regions only; it cannot establish coverage in China or a genuinely unseen region.

## Fixed split and methods

- Ten external sources are the outer units. Hold out one entire source at a time. For each of the seven fixed V50 `outer` point/risk functions, the other nine sources supply all calibration scores and region-level training labels; no held-source response may enter the threshold or regression.
- Use 90-day lag support and the already protocol-bound, hash-verified monthly full-mix feature. Outcomes are the actual held source's eligible hourly labels. Do not open any target-role `outer__outer` package or scoring file.
- Primary point/risk arm: `Scale-anchor + MCStd`, B0. Compare: (1) global equal-region 0.9 empirical score threshold fit on the nine training sources; (2) a one-feature unregularized linear regression trained on the nine source-local p90 scores, predicting a nonnegative held-source threshold; and (3) fixed raw anchor factor 2. HETScale + global equal-region threshold is a prespecified strong conditional-risk comparator.
- No bandwidth/hyperparameter search, binning, source exclusion, threshold clipping beyond `max(0, prediction)`, or feature selection. The local linear rule is an empirical comparator only, not a calibrated guarantee. Do not use held-source labels to select its threshold.

## Metrics and interpretation

Report source-macro and per-source hourly coverage, width, interval score, and lower/upper miss rate across outer functions. Show all ten source-fold outcomes and the q predictions. Regions, not hours, are the independent units. These ten regions were previously used in development; no confidence/significance or new-domain guarantee is claimed. Do not call an accuracy association an interval gain. No method freeze or unseen-region access follows from this experiment alone.

## Access limits

Read only the ten `external_labels` and 70 sealed external prediction archives needed for this cross-validation, plus hash-bound full-mix source features and seals. Do not read targets, proper archives, PACE/LGEE, 2025 external responses, or unseen sources. Do not alter any V50 artifact.

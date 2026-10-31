# V50: nested source-only shrinkage of monthly conditional interval thresholds

Registered 2026-10-27 before recomputing intervals. This is a standard nested-selection diagnostic following the negative direct monthly conditional-q interval result. It asks whether shrinking a feature-predicted source threshold toward a global equal-source conformal threshold can avoid the undercoverage of the unshrunk conditional rule. It is not presumed novel.

## Fixed data and outer evaluation

- Use the frozen V50 70 outer×external prediction archives, ten external labels, 90-day support, and the previously sealed legal full-month-mix proxy. Verify every original V50 and parent-probe hash. Do not open targets, PACE/LGEE, 2025 responses, or unseen regions; do not refit point/risk heads.
- Outer folds hold out one of ten entire external regions. For each of seven point/risk outer functions, all threshold prediction, shrinkage selection, and calibration use only the other nine regions. The outer-held region's labels are used only for final scoring.
- Primary arm is B0 `Scale-anchor + MCStd`; comparisons are outer global equal-source MCStd, direct monthly conditional MCStd (`lambda=1`), standard globally calibrated HETScale, and the fixed raw-anchor ×2 interval.

## Threshold model and nested lambda selection

For any training-source set S, compute `q_global` using the existing equal-source/hour 90th percentile of normalized MCStd scores. For each source in S compute its source-hour score p90. Fit the fixed ordinary least-squares intercept + one legal monthly EF-mix feature model to these source p90 values; predict `q_local` for the validation/test source, clipping only below at zero as in the existing comparator.

Use the prespecified shrinkage grid `lambda ∈ {0, .25, .50, .75, 1}` and `q_lambda=(1-lambda)*q_global + lambda*q_local`. For every outer-held source, choose lambda using an **inner leave-one-source-out split over only the nine outer-training sources**. Each inner fold recomputes both q values using its eight training sources, then scores the one inner-held source. Select the lambda with lowest equal-source mean 90% interval score among those whose inner source-macro empirical coverage is in [90%,95%] and whose worst inner-source coverage is at least 85%. If no lambda passes those fixed coverage constraints, select lambda=0. Ties choose the smaller lambda. No outer-held outcome can influence the selected lambda.

This nested rule intentionally permits fallback to the global threshold. It is compared against the unshrunk local rule, global threshold, HETScale, and raw anchor. No search over features, support, q-level, coverage gates, thresholds, or shrinkage grid is allowed.

## Reporting and interpretation

Report the lambda selected for every outer×held-source×outer-function, inner feasibility counts, per-source coverage/width/interval score/tail misses, source-macro metrics, and all hashes. Regions are the outer units; hours and seven repeated functions are not independent domains. A positive development result still requires a genuinely unseen-source confirmation under a separately frozen method and cannot establish China transfer.

The monthly proxy uses a project fixed-EF convention and simulated V50 release timestamps, not verified Chinese accounting factors or original Chinese publication vintage. The test is a standard nested shrinkage control, not an algorithm novelty claim. Preserve all prior results and refuse overwrite.

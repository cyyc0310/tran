# V147 — Source-learned conditional scales for nested risk curves

Frozen after V146 development outputs, before V147 target scoring. V146's global horizontal displacement is often driven by GRID source months and expands high-coal target risk curves excessively. Test whether source-only conditional scale prediction repairs this without making a special novelty claim. Give the ordinary vertical-hull family exactly the same learner and information.

## Budget and separation

Retain V144 canonical monthly fuel margins and V145's PACE/LGEE/SC months, two 30-fit/12-calibration rosters, alpha 0.10/0.20. No new identities or time windows. No target-hour trace, endpoint, factor floor, capacity or weather in features. Use the archived V145 common monthly features and its 20 response features for each of the two thresholds. The archived fit-region query features already remove their own whole region from the source bank; calibration/target features use the fixed 30-region bank.

For each proper-fit month, compute the V146 horizontal and vertical nested families using the 29 other proper-fit regions. Invert each family jointly against both factor endpoints and both thresholds to obtain a training displacement. These training labels use source hourly responses only. Reuse saved V146 curves and inverse labels for the separate calibration regions and saved label-free target curves. Fit/query invariance between the 29- and 30-region banks is an approximation and must be disclosed.

## Frozen estimators and score

For each family train two ordinary gradient-boosted conditional quantile regressors at q=0.50 and q=0.90: 150 trees, learning rate 0.05, maximum depth 2, minimum leaf 15, random seed 20261028; equal total weight per fit region. The nonnegative scale prediction is clipped to [1/256, 1]. A finite source inverse displacement trains directly; unsupported source months train at 1 and are separately counted. The two quantiles are sensitivity controls, not selected by target performance.

For each calibration month, score its actual inverse displacement divided by the independently trained scale. Unsupported months retain infinite score. Take the maximum within each of 12 calibration regions and the same finite-sample rank as V146. At target composition evaluate the curve at the smallest grid displacement at least radius times predicted scale. A product above 1, or an infinite radius, returns the actual-margin physical hull; record this fallback. A product of zero uses grid index zero. This adds a nested full-hull endpoint beyond the horizontal grid and cannot narrow the family at the transition.

## Evaluation

Four methods: horizontal/vertical family × median/q90 scale. Report all methods, both seeds, alpha values and per-region results; compare with V146 joint unscaled families and V145 archived per-threshold anchors, recomposition and residual/direct CQR. Report joint two-threshold coverage, individual endpoint coverage, width and the fixed 90% robust interval score. These are monthly tail-count endpoints under an assumed factor box, not hourly forecast intervals.

Assertions: archived input hashes; profile alignment; fit/cal/target disjointness; every fit curve uses exactly the other 29 fit regions; delta-zero identities; curve nesting; scale feature whitelist; complete fit/cal/target denominators; integer bounds inside physical hull; alpha nesting. Save fit inverse labels, feature rows, donor provenance, predicted scales, calibration scores/radii and output hashes. No V146 or V145 artifacts are mutated. A successful scale predictor remains an established heteroscedastic conformal baseline; further attribution and literature review are necessary for a methodological innovation claim.

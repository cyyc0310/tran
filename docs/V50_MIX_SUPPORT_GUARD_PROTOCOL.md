# V50 monthly-mix correction with a source-support guard

Registered 2026-10-27 before this computation. The fixed correction strength is λ=0.75, chosen as a development candidate from the already reported shrinkage path; that prior selection means this round is still development evidence, not confirmation.

## Fixed rule

- Reuse the same frozen V50 seals, strict per-query-date fuel feature, 10 sources, seven outer functions, 90-day eligible labels, Scale-anchor paths, and factor-2 intervals. Recheck all hashes.
- In each outer × held-source fold, estimate the equal-source-weighted intercept-plus-mix OLS using the other nine sources, as before. Define the feature support interval from those nine sources' eligible query-date feature values only: `[min(train_feature), max(train_feature)]`.
- For each held-source day, apply `0.75 × predicted_offset` only when that day's feature lies within the training support interval. If it lies outside, apply zero offset (the original raw anchor). This is a fixed applicability-domain fallback; do not tune the support interval, λ, or fallback from held-source labels.
- Compare zero shift, global shift, unguarded 0.75 mix shift, and guarded 0.75 mix shift. Report source-level metrics and fraction of days falling back, plus hourly particle clipping. No target/unseen input, no point/risk refit, and no new feature.

## Interpretation boundary

This is still a development check on ten reused sources. The λ was selected from those sources, so only a future frozen genuinely unseen-source run can provide an independent transfer check. A passing guard experiment does not by itself establish method novelty, China deployment, or a coverage guarantee.

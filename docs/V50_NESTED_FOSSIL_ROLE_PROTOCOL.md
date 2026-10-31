# V50 nested source-held-out selection for fossil-role correction

Registered 2026-10-27 before computation. The previous fixed shrinkage path showed an apparent λ=.25 gain but evaluated all λ values on the same ten outer sources. This round uses nested source validation so the held outer source has no influence on λ selection.

## Fixed algorithm

- Reuse only the same frozen V50 prediction/calibration seals, legal three-month query-date composition features, 10 external sources, seven outer functions, 90-day eligible labels, Scale-anchor paths, and factor-2 intervals. Recheck hashes; no target/unseen source or point/risk refit.
- For each outer function and outer held source, use the other nine sources as the inner training pool. For each inner-held source, fit the EF-only daily residual model and the orthogonal fossil-share residual increment on the remaining eight sources, with equal total source weights. Score fixed λ∈{0,.25,.50,.75,1} on the inner-held source's own held labels. No outer-held data may enter any inner step.
- Select λ by minimizing the mean source-level 90% interval score over inner-held sources, subject to mean source coverage ≥90% and minimum source coverage ≥85%. If no λ satisfies both constraints, choose λ=0 (EF-only fallback). Refit the selected EF+orthogonal fossil increment on all nine outer training sources and apply the selected λ to the outer-held source.
- Compare nested selection to raw anchor, global bias, and unshrunk EF-only baseline. Report outer-held chosen λ, all per-source CRPS/level/coverage/width/interval-score/tails/clipping, and the inner selection diagnostics. Independent unit remains region, not day or outer function.

## Interpretation boundary

This is nested validation within ten previously used development sources, not an independent external confirmation. It tests whether a training-only rule can select conservative fossil-role strength; it cannot establish China transfer, guarantee coverage, causal dispatch interpretation, or scientific novelty. No unseen source is opened.

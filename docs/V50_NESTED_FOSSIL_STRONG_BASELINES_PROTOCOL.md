# Nested fossil-role correction across sealed V50 point baselines

Registered 2026-10-27 before computation. The previous nested fossil-role probe used only the Scale-anchor particle paths. This round tests whether the same monthly correction adds value to the sealed B0/B1 point baselines.

## Fixed design

- Reuse the V50 freeze/seals, 70 external prediction packages and ten external labels, strict query-date three-month fuel composition features, 90-day support, and all five frozen arms: Scale-anchor, B0 SizeET ReplaceR, B0 SizeET AlignedKeep, B1 SizeCapRidge ReplaceR, and B1 SizeCapRidge AlignedKeep. Verify file and seal hashes. Never read target/unseen-role predictions or scores; no point/risk refits.
- For each outer point function × arm × outer held source, derive daily residuals from that arm's own source predictions. On the other nine sources, conduct inner leave-one-source-out selection over λ∈{0,.25,.50,.75,1} for the orthogonal fossil-share increment beyond the EF-only intercept/slope model. The selection rule is unchanged: minimum mean inner-source factor-2 interval score among candidates with mean coverage≥90% and worst inner-source coverage≥85%; if none qualify, λ=0.
- Refit on the nine outer training sources and evaluate only on the outer-held source. Compare raw arm, equal-source global shift, EF-only correction, unshrunk fossil increment, and nested-selected fossil increment under the same factor-2 interval. Report each arm separately and aggregate equally by source after summarizing seven outer functions within source.

## Interpretation boundary

This is source-held-out development evidence on ten reused sources, even though the inner selection protects each outer source from coefficient choice. The five arms are frozen point predictors; this is not a comparison to all V50 risk methods or an independent region confirmation. No novelty, China transfer, or coverage guarantee may be claimed from this run.

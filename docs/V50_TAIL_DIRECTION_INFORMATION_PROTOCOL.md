# Development-only test: can monthly mix predict held-source tail direction?

Registered 2026-10-27 after the frozen TVA candidate failed. This diagnostic returns only to the same ten repeatedly used V50 development regions. It does not reopen TVA, any other target, or any new region; it cannot repair or replace the TVA result.

## Question and fixed data

Can three legally available, time-local monthly composition summaries predict which side of the interval will miss more often on an entirely held-out source? Use the already sealed V50 monthly tables and label support dates, plus the precomputed source-level rows from the V50 cross-fitted asymmetric-risk run. No hourly labels or prediction packages are reopened by this probe.

- Outcome per source: mean lower miss rate minus mean upper miss rate for B0-SizeET-AlignedKeep + MCStd + EF-only location shift + symmetric 90% calibration, averaged over the seven outer functions. Positive means relatively more lower-tail misses; negative means relatively more upper-tail misses.
- Fixed predictors per source: mean of the query-date three-month EF proxy, mean three-month fossil share, and mean three-month wind+solar share, computed only on the source's V50 90-day eligible query dates. Each feature is tested in its own one-variable linear regression. No multivariate model, transforms, interactions, feature selection, or target-specific tuning.
- Evaluate leave-one-source-out MAE and RMSE against the fold training-mean predictor, plus held-out direction accuracy against zero and the training-mean sign. Source is the unit; n=10. Report every fold and feature. Do not infer significance from hourly counts.

## Interpretation

This is a descriptive diagnostic with ten old development sources and three predeclared low-dimensional feature checks. Better LOO scores would indicate a hypothesis for later development only, not justify changing the failed TVA intervals or promise a new-region gain. Null or worse scores argue that monthly composition alone cannot safely determine which tail to tighten. No method is frozen and no target/unseen values are opened.

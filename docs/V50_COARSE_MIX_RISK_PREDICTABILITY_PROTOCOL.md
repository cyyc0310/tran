# V50 low-frequency mix → hourly-risk predictability probe

Registered 2026-10-27, after the source-influence diagnostic. This is a development-only information-value probe. It does not alter V50 and cannot trigger access to an unseen target.

## Question and scope

Can a legally available, low-frequency generation-mix summary predict which source regions have larger standardized hourly prediction errors? This tests whether monthly fuel composition could help identify *risk heterogeneity* before local hourly CIF is observed. It is not an interval method and does not claim coverage.

Use only the same ten external 2024 calibration regions, their sealed V50 70 outer×source predictions, the ten sealed external labels, and the already sealed calibration files. Do not open target-role `outer__outer` archives, target truth/scoring tables, proper archives, PACE/LGEE, 2025 external responses, or unseen-region data. Do not refit point or risk predictors. Fit only ten deterministic one-feature diagnostic regressions as part of the registered outer-region cross-validation.

## Fixed estimand and information budget

- **Hourly-risk response:** for each external region and each of seven frozen point/risk outer functions, calculate the empirical within-region p90 of hourly normalized interval scores using `Scale-anchor + MCStd`, B0, 90-day response lag. The source-level response is the median of these seven outer-specific p90 values. Use p99 only as a named tail sensitivity, not for model selection.
- **Low-frequency predictor:** mean of the `POS` share (the visible positive-emission share) over the three latest fully released monthly contexts, averaged across that region's fixed July–December query dates. Read only `context_g[..., 0]` from sealed prediction archives. No monthly response newer than each query's registered context may be introduced.
- **Information comparison:** a constant predictor (training-region mean) versus one unregularized linear model with intercept and the single registered mean-POS feature. For each of ten folds, hold out one entire region, train on the other nine, predict the held source. No hyperparameter search, feature screening, stratification, source removal, or target-wise calibration.
- **Metrics:** held-region MAE and RMSE for p90 are primary; p99 MAE/RMSE and fold-wise slope sign are secondary. Fit ten source-held-out one-feature regressions for each response (20 diagnostic fits total). Report all ten held-source predictions, feature values and outcomes. Regions, not their hours or seven shared outer functions, are the units for inference.

## Interpretation gate

Ten previously used source regions are too few for a reliable generalization or significance claim. Any improvement is only a hypothesis-generating signal and must be compared in a later preregistered experiment against simple scale anchoring and strong conditional-risk baselines under the same public monthly-information budget. A null or unstable result argues against using monthly POS share to narrow intervals. Neither outcome authorizes source deletion, threshold tuning, algorithm freeze, or opening a genuinely unseen region.

## Audit

Before reporting probe metrics, reproduce the fixed hourly score formula and verify the V50 source package and label hashes and their 90-day support. The script writes a source-level table, summary, and input/script hashes to a new directory; it refuses to overwrite an existing run.

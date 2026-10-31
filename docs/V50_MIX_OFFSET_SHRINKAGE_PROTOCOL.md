# V50 monthly-mix offset shrinkage path

Registered 2026-10-27 before running. Motivation: the unshrunk source-held-out mix correction improves average proper scores but produces severe SEC undercoverage and additional clipping. This round measures a fixed shrinkage path; it does not select a best coefficient from held-source performance.

## Fixed design

- Reuse the sealed V50 source-only inputs and strict query-date mix feature from `V50_DAILY_ASOF_MIX_LEVEL_PROTOCOL.md`; revalidate every seal and hash. Use the same seven outer functions, ten held sources, 90-day eligible support, Scale-anchor raw paths, factor-2 interval, and [0,1000] bounds.
- Fit the same equal-source-weighted source-LOO intercept-plus-mix OLS correction using the other nine sources. Apply fixed coefficients λ∈{0.25, 0.50, 0.75, 1.00} to its predicted daily offset: `shift = λ × offset`. The already computed zero-shift and global-shift baselines are retained. No per-held-source selection, tuning, new feature, point/risk refit, or target-score access.
- Report every fixed λ by source and macro for hourly/daily CRPS, daily-level MAE/bias, coverage, width, interval score, tail miss direction, and clipping frequency. Compare SEC and SCL explicitly; source remains the independent unit.

## Interpretation boundary

This is a shrinkage sensitivity curve on ten previously used development sources. It can diagnose the bias/coverage tradeoff but cannot establish a chosen deployable λ or guarantee coverage. A future selection rule would require nested source-held-out tuning using only training sources and a genuinely unseen-source confirmation. No unseen data is opened in this experiment.

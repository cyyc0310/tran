# V50 monthly fuel composition: carbon intensity vs dispatch-role features

Registered 2026-10-27 before computation. Prior experiments found a signal in one fixed-EF intensity proxy, but the proxy conflates fuel carbon content with system composition. This experiment tests whether low-dimensional monthly fuel groups explain source-held-out daily level bias beyond that scalar.

## Fixed feature construction

- Reuse the frozen V50 seals, two frozen monthly proxy tables, ten external sources, seven outer functions, 90-day eligible labels, Scale-anchor paths, and factor-2 intervals. Verify all hashes; no target/unseen source and no point/risk refit.
- At each query date, use the same three latest complete monthly records with `period_end` and `available_at` no later than the query date. Average their eight shares and fixed-EF proxy. Build groups from those same rows: `fossil = coal + gas + oil + other`; `variable_renewable = wind + solar`; `nuclear = nuclear`; hydro is the omitted reference group. All features are defined even when fossil generation is zero; do not use within-fossil ratios.
- For each outer × held source, fit source-equal weighted OLS on other nine sources' daily mean residuals. Standardize each model's features using only the outer training rows with equal total source weight. Compare fixed models: EF scalar; EF + fossil share; EF + variable-renewable share; EF + fossil + variable-renewable; and grouped composition (fossil + variable-renewable + nuclear, with hydro omitted). No feature/model selection based on held-source performance.
- Apply each learned daily offset uniformly to all 128 hourly particles, with the same clipping/interval procedure. Report source-macro and per-source CRPS, level MAE/bias, coverage, width, interval score, and clipping. Source remains the independent unit.

## Interpretation boundary

This is a fixed low-dimensional feature ablation on ten previously used development sources. Coefficients are descriptive transfer regressions, not identified causal dispatch effects. It can reveal whether group structure adds robust predictive information in these proxies; it cannot establish China data equivalence, unseen-region transfer, a coverage guarantee, or novelty.

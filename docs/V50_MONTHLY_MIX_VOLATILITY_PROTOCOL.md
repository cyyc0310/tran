# V50: test whether month-to-month mix volatility predicts held-source hourly risk

Registered 2026-10-27 before recomputing the monthly features. This is a development-only source-held-out information-value probe. It follows the positive but small-sample V50 full-mix level signal and tests one distinct, prespecified hypothesis: rapid legal monthly mix changes may mark operating instability that a three-month mean level misses.

## Fixed inputs and response

- Reuse only the frozen V50 ten-source external monthly inputs and the already sealed V50 full-mix probe's source-level outcomes. Do not read target-role packages, 2025 labels, PACE/LGEE, or unseen-region data; do not refit production point or risk heads.
- Response is the existing source-level median across seven frozen outer functions of the `Scale-anchor + MCStd`, B0, 90-day normalized hourly-score p90. p99 is secondary. Each of the ten sources is one unit; repeated outer functions, dates, and hours are not independent units.
- At each registered July–December 2024 query date, select exactly the three latest complete-calendar monthly records that had been released by that date, using V50's frozen period-end and available-at rules. Convert each month's eight shares to the fixed-EF monthly generation proxy `[1000,400,800,0,0,0,0,400]`.
- **Level feature:** average the three-context EF proxy over the registered query dates, reproducing the prior full-mix feature.
- **Volatility feature:** for each query date, calculate the mean of the two absolute successive changes across its three ordered contexts; then average across query dates. Units are g/kWh of the fixed proxy. No smoothing, feature selection, alternate window, or threshold search.

## Fixed source-held-out comparison

Fit ordinary least squares in ten leave-one-source-out folds, separately for p90 and p99, comparing four fixed feature sets: (1) constant mean; (2) intercept + level; (3) intercept + volatility; and (4) intercept + level + volatility. Standardize nonconstant training features using the training fold's mean and standard deviation before OLS; if a feature has zero training variance, that fold falls back to an intercept-only prediction and is marked. No regularization or model selection. Report every held-source prediction, fold coefficient, feature value, MAE, RMSE, and sign accuracy for deviations from the training-source mean.

The primary comparison is whether adding volatility to level improves p90 LOO MAE and RMSE versus level-only. A secondary comparison is volatility-only versus the constant baseline. p99 is descriptive secondary. No hourly interval, coverage, or CIF forecast is produced by this probe.

## Interpretation boundary

Ten previously used source regions make this hypothesis-generating only. Any improvement is not evidence of calibration, a novel algorithm, China transfer, or a deployment-ready feature. A positive result may justify a separately frozen matched-budget interval experiment with the already established strong baselines; it does not itself authorize that method to be frozen or any new target opened. A null or worse result closes the volatility-feature branch unless genuinely new legal information becomes available.

The monthly proxy uses a project fixed-EF convention and simulated V50 release timestamps, not verified Chinese accounting factors or original Chinese publication vintage. This is explicitly outside the user's China data validation. Refuse overwrite; bind protocol, code, V50 freeze, parent source outcomes, and monthly input hashes before writing outputs.

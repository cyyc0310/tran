# V50 within-POS monthly mix → hourly-risk probe

Registered 2026-10-27 after the predeclared single-POS probe did not improve held-source risk prediction. This new probe asks whether fuel composition **inside** the positive-emission aggregate carries more useful risk information. It remains an exploratory information-value test on previously used U.S. development regions, not a China validation.

## Frozen data and feature

- Outcome: reuse the previously sealed source-level p90 normalized hourly-risk response for `Scale-anchor + MCStd`, B0, 90-day support. Its source table and hash must match the completed prior probe summary.
- Input: the two external monthly CSV files already hash-bound in V50 `freeze.json`; filter to the fixed ten external regions and use only rows whose full monthly period and release timestamp precede each July–December 2024 query date.
- For each query date, take the three latest legal monthly records. Compute the POS-weighted emission factor from the four public positive-emission shares: `(1000*coal + 400*gas + 800*oil + 400*other)/(coal+gas+oil+other)`. The single registered predictor is the mean of this quantity over available query dates and their three contexts. Require positive POS mass; fail closed if any source lacks the registered support.
- This proxy uses the project’s existing EF convention for an information-value diagnostic. It does not certify the factors as Chinese official accounting factors or equate the proxy to observed CIF.

## Evaluation

Compare a constant training-source mean with one unregularized, intercept-plus-single-feature linear regression. Use ten leave-one-region-out folds, no tuning, source screening, or additional predictors. Report p90 held-source MAE/RMSE as primary, p99 as secondary, every source prediction, and fold-slope direction. The ten regions are the evaluation units. Do not treat hours, months, three contexts, or seven outer functions as independent regions.

No target-role archive, target 2025 label, proper archive, PACE/LGEE, or unseen-region data may be opened. Do not change the V50 score or calibrator, do not fit production point/risk models, and do not use the diagnostic to select/exclude sources or thresholds. A positive result is only a hypothesis requiring later same-budget comparisons against simple anchoring and strong risk baselines; a negative result does not show that every monthly fuel statistic is uninformative.

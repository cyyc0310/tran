# V50 full monthly mix as a cross-source level correction

Registered 2026-10-27. This tests an unexamined component of the information budget: whether legally available, detailed monthly fuel composition can predict **source-wide level bias** in the frozen hourly anchor. The hourly shape is held fixed except for the same additive shift to all hours. This is a development source-held-out probe, not China validation or a new-domain confirmation.

## Fixed data and response

- Ten external regions, each held out in turn; seven already frozen V50 outer point functions per region. Labels are only the 90-day-eligible external hourly responses already sealed in V50.
- Frozen feature: the hash-verified full monthly EF-weighted mix proxy already computed from three latest fully released monthly contexts. No feature selection or alternate lag.
- Primary response for each training source/outer pair: mean over eligible source days and hours of `truth − mean(Scale-anchor raw paths)`. This is a single source-level signed level bias, not an hourly residual model.
- Target source receives no local hourly labels during fitting. The feature regression is trained on the other nine source-level biases, with an intercept and one feature. The simple correction is the mean bias of those same nine sources.

## Fixed methods

1. No correction.
2. Global mean source-bias correction (nine training sources).
3. One-feature monthly-mix regression correction (same nine sources).
4. Raw anchor factor-2 intervals as the fixed interval-efficiency reference.

The correction shifts every hourly path by the held-source predicted offset and clips to [0,1000]; it does not reshape paths, refit the anchor, or recalibrate risk. The raw factor-2 interval is recomputed from each corrected path distribution using the original 128-draw empirical endpoints. Report both point-distribution CRPS and 90% interval coverage/width/interval score, with per-source outcomes. Also report daily-mean CRPS to separate level prediction from hourly shape.

## Audit and limits

Verify the V50 seals, all 70 external prediction-package hashes, ten label-package hashes, fixed 90-day support, and the prior full-mix feature-table hash. Do not open target-role packages, target response/scoring tables, proper archives, PACE/LGEE, 2025 external responses, or unseen-region data. Do not refit production point/risk models. The ten regions were previously used in development; source-held-out improvements are hypothesis-generating only and cannot trigger method freeze or unseen-region access.

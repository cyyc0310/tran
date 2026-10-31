# V50 full monthly fuel mix → hourly-risk probe

Registered 2026-10-27 after two diagnostics: (1) a single positive-emission share did not improve held-source risk prediction, and (2) the preregistered within-POS intensity probe correctly failed closed because some legal monthly contexts had zero POS mass, making a POS-conditional factor undefined. This round uses the full monthly fuel mix, including the amount of POS generation, so the predictor is defined for zero-POS months.

## Question and fixed inputs

Does the publicly available monthly fuel composition, reduced to the project's current fixed-EF monthly generation proxy, predict region-level standardized hourly risk out of source?

- Keep the same ten external calibration regions and the same already sealed source-level response: median across seven outer functions of each source's hourly-score p90 (p99 secondary), for `Scale-anchor + MCStd`, B0, 90-day support.
- Use only the two hash-bound V50 external monthly CSVs. For every July–December 2024 query date, select the three latest fully released, full-calendar months using the same period-end and release-cutoff rules as the V50 context builder.
- Compute the coarse proxy `sum_f EF_f * monthly_fuel_share_f`, with the existing registered factors coal 1000, gas 400, oil 800, nuclear/hydro/wind/solar 0, other 400. Average over each query date's three contexts and then over query dates. This is a monthly generation-side proxy, not observed CIF or a verified China accounting factor.
- Compare the mean-only null predictor with one unregularized intercept-plus-one-feature linear regression in ten leave-one-region-out folds. No tuning, screening, source removal, or extra features.

## Scope and reporting

Report p90 LOO MAE/RMSE as primary, p99 secondary, every held-source prediction, fold slope direction, and hashes/support. The ten regions are the units; hours, months, contexts, and repeated outer functions are not independent. No target-role packages, target 2025 response, PACE/LGEE, or unseen regions may be read. No production-model refits or V50 modifications. Any signal is hypothesis-generating and must later beat simple anchoring and strong conditional-risk baselines at the same information budget before it can support an algorithm claim or unlock unseen-region confirmation.

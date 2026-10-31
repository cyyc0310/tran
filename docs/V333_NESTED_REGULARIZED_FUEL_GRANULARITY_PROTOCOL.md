# V333 protocol: nested regularization of detailed vs coarse fuel mixes

Registered before running on 2026-10-30. V332 found that unregularized eight-source Ember NNLS loses to five-group NNLS under whole-region cold start, although it helps Northwest and Northeast. V333 checks whether overfit in the higher-dimensional fuel schema explains this result.

## Frozen cohort and candidates

Reuse V329's seven region blocks and 2022/2023 target folds, with the same 2021–2022 MEE labels and same lagged Ember mix table. Fit both the eight-source and five-group Ember feature schemas using NNLS and ridge-NNLS. Select λ separately for each schema in the outer training data using V330's nested spatial rule and fixed candidates `{0, 0.001, 0.01, 0.1, 1, 10}`. For 2022 outer targets, tune by leaving each available training region out of 2021; for 2023 targets, validate 2022 rows using only 2021 labels and leave each validation region out. Select the smallest λ within 1e-12 of minimum MAE.

Also report the V332 raw schemas and V329 NBS five-group score as context. No held-region target label may enter fitting or selection.

## Outcomes and decision

Report pooled and region-equal MAE/RMSE/P90, region-year folds, selected λ, and a region-cluster bootstrap for paired Ember full8-ridge minus Ember coarse5-ridge error. Fuel detail is retained only if regularized full8 shows gains in at least four of seven regions, both target years, and its paired bootstrap interval favors full8 without a tail-error penalty.

## Scope

This is a model-dimension control on retrospective estimated Ember mixes and annual consumption-side MEE labels. It does not validate user data, hourly CIF, or coverage. Ridge is a standard estimator, not a novelty claim.

# V330 protocol: nested spatial regularization for cold-start annual factor transfer

Registered before running on 2026-10-30. V329 found that nonnegative composition regression transfers better than a training mean but can overpredict in Northeast and Northwest; some fitted wind coefficients exceed 1.14 kgCO₂/kWh, suggesting weak group-factor identification. V330 tests whether train-only, spatially selected ridge regularization stabilizes this transfer.

## Frozen data and outer evaluation

Use the exact V329 target table, NBS/Ember features, seven geographic blocks, province cohort, and outer prediction folds. For each outer fold, exclude every training row from the held region and use only target labels earlier than the test year. The held region's outcomes never enter fitting or tuning.

## Candidate models

- NBS NNLS: reproduce the V329 five-group nonnegative least-squares model.
- NBS ridge-NNLS: minimize `||Xβ−y||² + n λ ||β||²` subject to `β ≥ 0`, where `n` is the inner-fit row count.
- Outside-region training mean and Ember NNLS are unchanged baselines.

Freeze `λ ∈ {0, 0.001, 0.01, 0.1, 1, 10}`. For each outer fold, select λ by inner region-held-out validation using only the outer training region set. For a 2022 outer target, inner validation leaves one training region out of 2021 labels, since no earlier MEE year is available. For a 2023 outer target, validate 2022 rows using only 2021 labels and leave each inner region out. Pick the smallest λ within 1e-12 of the lowest pooled inner MAE. If an inner fold lacks training rows, skip it and record counts.

## Outcomes and decision rule

Report outer pooled and equal-region MAE/RMSE/P90, region-year folds, paired region-cluster bootstrap differences, selected λ, and coefficient ranges. A candidate increment requires lower pooled MAE than V329 NBS NNLS, improvement in at least four of seven regions, no material P90 degradation, and directionally consistent improvement in both target years. The current seven-region cohort is exploratory and has low cluster power.

## Scope

Targets are annual MEE consumption-side factors, not hourly generation CIF. This tests a standard regularization control, not a novel algorithm by itself. Any positive result is only a robustness component that still needs independent time and hourly validation.

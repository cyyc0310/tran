# V328 protocol: scope-aware NBS mix transfer with an explicit unobserved share

Registered 2026-10-30 before V328. V327 found that actual NBS-derived monthly five-category mix preserves the annual LOPO transfer signal, but the monthly NBS table covers only above-designated-size enterprises. V328 tests whether carrying the known annual all-plant minus monthly reported-generation share as an explicit sixth category improves transfer.

## Fixed data and split

Reuse the V327 official MEE annual target, NBS monthly categories, and Ember five-group control; use V320's same-provider annual all-plant generation table for the same prior feature year. Retain the exact V327 target years (2022, 2023), lagged feature years, 30 provinces, LOPO folds, NNLS-without-intercept model, and mean/reference baselines.

For each feature year and province, sum February cumulative Jan–Feb once plus March–December. Construct normalized composition shares from the five NBS groups; reconstruct missing Tianjin hydro only as the unresolved remainder of reported monthly total. Let coverage be `monthly above-size total / annual all-plant total`, constrained to `[0,1]` with any overage beyond rounding treated as a data-contract error. The six model features are five composition shares multiplied by coverage plus a sixth `unobserved remainder share = 1 − coverage`; the six features sum to one. Fit a separate nonnegative coefficient for the unobserved share. Do not infer its technology composition from the MEE held-out target.

Compare scope-aware NBS-six NNLS against V327 normalized NBS-five NNLS and same-split lagged Ember-five NNLS. Do not change folds, groups, or coefficients based on test outcomes.

## Outcomes and decision

Report MAE, RMSE, P90 absolute error by target year and pooled 2022/2023, province-cluster bootstrap intervals for paired absolute-error differences, province counts improved, and severe tail errors. The scope-aware branch advances only if it improves on V327 NBS-five in both target years and pooled metrics without worsening the 90th-percentile tail materially. Otherwise reject an explicit common coefficient for the unobserved share as insufficient evidence. This is annual consumption-factor transfer, not hourly CIF validation.

# V332 protocol: fuel granularity under whole-region cold start

Registered before running on 2026-10-30. V300 found no overall LOPO advantage for eight Ember source groups over the corresponding five coarse generation groups in the single 2023 target. V329 found region-level cold-start failures and no robust NBS advantage over Ember coarse groups. V332 tests whether more detailed fuel groups help when entire Chinese statistical regions are held out.

## Frozen cohort and models

Reuse V329's seven regions, 30 MEE provinces, 2022/2023 forward targets, and target-year training restrictions. For each held region, exclude every province in that region from model fitting. Use Ember's same lagged annual mix snapshot for both feature schemas:

- Ember full8: coal, gas, other fossil, bioenergy, hydro, nuclear, wind, solar;
- Ember coarse5: thermal (sum of coal, gas, other fossil, bioenergy), hydro, nuclear, wind, solar.

Fit nonnegative least squares without intercept, with no hyperparameter tuning. Include the V329 outside-region mean and NBS-reported five-category NNLS as context baselines. The latter uses a different data product and is not a same-source feature-granularity contrast.

## Outcomes and decision

Report pooled province-weighted and equal-region MAE/RMSE/P90; each region-year fold; region-cluster bootstrap for paired full8-minus-coarse5 absolute error; and the number of regions improved. Detail adds predictive value only if full8 has consistent gains across both years and several regions, without reducing only one severe failure while causing broad degradation elsewhere.

## Scope

Ember is a retrospective, partly estimated mix snapshot without verified historical release vintages. Targets are annual MEE consumption-side factors, not generation-side hourly CIF. This experiment isolates only fuel granularity under geographic block transfer; it does not test the user's source table or hourly shape/coverage.

# V327 protocol: official monthly generation mix for cold-start annual EF transfer

Registered 2026-10-30 before the V327 run. V300/V301 showed that lagged Ember mix predicts held-out provincial annual MEE factors better than a cross-province mean, but it used a retrospective estimate of generation mix. V327 tests whether an NBS-derived public monthly mix carries the same transfer signal.

## Question

For a province with no prior local MEE factor, can its previous-calendar-year NBS-derived monthly generation categories improve prediction of the next official annual provincial grid CO₂ factor, under leave-one-province-out (LOPO) evaluation? How does it compare with the same-split lagged Ember five-group baseline?

## Data contract and frozen split

- Target: V300 transcription of official 2021–2023 MEE provincial average electricity CO₂ factors. This is an annual consumption-accounting factor, not hourly generation CIF.
- Candidate predictor: V320 OREO/NBS-derived monthly province generation in thermal, hydro, nuclear, wind, and solar. February is the Jan–Feb cumulative report; sum February once plus March–December. Monthly scope is above-designated-size enterprises, not all plants.
- Missing Tianjin hydro is not set to zero. Define its fifth category as the unresolved remainder `max(total − thermal − nuclear − wind − solar, 0)`, then normalize the five groups to sum one. For provinces with hydro, use the reported hydro category; normalize the five reported groups to one to absorb small rounding differences. Record the Tianjin treatment and all category shares.
- Lag: target year `y` uses month-table year `y−1`. Evaluate 2022 (train on 2021 target rows using 2020 features) and 2023 (train on 2021–2022 target rows using 2020–2021 features). The 2021 target trains only the 2023 model, using 2020 features. Hold out each target province from all training rows.
- Model: nonnegative least squares (NNLS) with no intercept, matching V300/V301's five-group model. Baselines: held-out-fold training-target mean and same-province previous-year official factor as an unavailable-data reference only. Compare the lagged Ember five-group NNLS on the exact same splits.

## Outcomes and audit

Report MAE, RMSE, P90 absolute error by target year and pooled across 2022/2023; paired province-cluster bootstrap interval for the NBS-minus-Ember absolute-error difference; number of provinces improved versus the training mean; per-province errors; and input/protocol/script hashes. Do not tune fuel grouping or category handling on held-out errors. A positive result establishes only annual-level historical transfer under the published table's scope; no hourly coverage, temporal shape, publication-vintage availability, or consumption-flow causality is inferred.

## Decision rule

The actual NBS feature is a meaningful data-path gain if it beats the same-split Ember proxy with a paired improvement whose province-bootstrap interval excludes zero, retains improvement over the training mean in both temporal targets, and avoids a severe regional tail-error increase. If not, reject public aggregate generation shares as a sufficient annual level anchor by themselves and identify the residual information gap.

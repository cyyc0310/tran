# V335 protocol: heat-rate feature under nested regularization

Registered before running on 2026-10-30. V334 found that lagged provincial standard-coal consumption rates improve raw five-group mix NNLS under LOPO and whole-region holdout, but gains are concentrated in Northwest/Northeast and seven-region intervals cross zero. V330/V333 also showed that ordinary ridge regression materially improves mix-only transfer. V335 tests whether the local efficiency field adds value beyond that stronger regularized baseline.

## Frozen folds and model schemas

Reuse V329's seven whole-region blocks and 2022/2023 forward target folds. Use the V334 NBS and Ember five-group features, plus the two lagged CEC rate variants (generation coal consumption, table 2-41; supply coal consumption, table 2-43). For each source, compare:

- mix-only five-group ridge-NNLS;
- rate-adjusted ridge-NNLS, where the thermal feature is `thermal_share × local_rate/300` and four nonthermal features are unchanged.

For every outer fold and schema, select λ separately from `{0, 0.001, 0.01, 0.1, 1, 10}` in inner whole-region validation using only outer-training labels. For a 2022 target, inner CV leaves a training region out of 2021; for a 2023 target, validate 2022 rows using 2021 training rows while leaving each inner region out. Choose the smallest λ within 1e-12 of minimum pooled MAE. No outer region labels enter model fitting or λ choice.

## Outcomes and decision

Report pooled and region-equal MAE/RMSE/P90, each region-year fold, selected λ, and paired seven-region bootstrap for adjusted-minus-baseline error within NBS and Ember. A heat-rate increment advances only if the adjusted model improves over its same-source mix-only ridge in both years, at least four of seven regions, with a paired interval favoring the adjusted model and no material P90 degradation.

## Scope

CEC 2024 is a retrospective vintage, so past values are not verified as-of forecasts. The target remains annual MEE consumption-side factors. Heat rate is standard-coal consumption, not CO₂ intensity; plant scopes differ; no hourly CIF or coverage claim is tested. Ridge is a standard baseline, not a novelty claim.

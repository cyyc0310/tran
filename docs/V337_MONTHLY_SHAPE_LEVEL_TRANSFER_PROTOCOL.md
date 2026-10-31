# V337 protocol: within-year technology shape beyond annual mix level

Registered 2026-10-30. The project aims to separate temporal shape from annual level. V329–V336 aggregate monthly technology output into an annual lagged composition and thus discard within-year shape. V337 tests whether a low-dimensional seasonal signature adds transferable signal to the same annual mix baseline.

## Features and folds

Reuse seven whole-statistical-region outer folds and 2022/2023 annual MEE targets. Retain the same five annual mean shares as V335. Add two first-harmonic coefficients (sine and cosine over calendar month) for each of the five technology groups, fit to within-year monthly generation shares. NBS February is Jan–Feb cumulative in the source audit, so exclude February from seasonal fitting; use March–December observations. Ember uses all twelve monthly observations. The annual mean baseline remains exactly V335's existing feature construction. No target-year or held-region labels inform features.

Compare mean-only ridge-NNLS to mean-plus-seasonal-shape ridge-NNLS. Select λ separately inside each outer fold and schema from `{0, .001, .01, .1, 1, 10}` with inner region-heldout MAE. Report pooled/yearwise and region-level MAE/RMSE/P90 and paired seven-region cluster bootstrap. Do not interpret harmonic coefficients causally.

## Gate and limits

Shape features advance only if they reduce region-equal MAE in both years, improve at least 4/7 regions, and the paired region-cluster bootstrap excludes zero without worsening yearwise P90. This is annual target prediction, not hourly CIF validation. With only seven clusters and two outer years, confidence intervals are low-power. Seasonal harmonics are a standard representation; any novelty would need to come from a better identified transfer/risk mechanism, not this feature basis alone.

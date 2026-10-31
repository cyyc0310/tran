# V330 results: train-only spatial regularization improves pooled cold-start error, but not consistently by region

Run date: 2026-10-30. Frozen design: [V330 protocol](V330_SPATIAL_NESTED_RIDGE_PROTOCOL.md). Outputs: `results/v330-spatial-nested-ridge-transfer-r1/`.

## Result

Nested ridge-NNLS selected only weak penalties (λ=0.001 in 9 outer region-year folds; λ=0.01 in 5). It improved pooled province-weighted NBS MAE from 0.07461 to 0.05144 kgCO₂/kWh (−31.1%); region-equal MAE moved from 0.07567 to 0.05450 (−28.0%). Pooled RMSE fell 0.11751→0.06482 and P90 absolute error 0.15165→0.11883. Improvements were directionally consistent in both target years: province MAE 0.08200→0.05591 in 2022 and 0.06722→0.04697 in 2023.

The apparent gain is concentrated in two regions that were severe failure modes. Northwest MAE fell from 0.19535→0.02946 in 2022 and 0.16079→0.02083 in 2023. Northeast fell from 0.11260→0.09014 and 0.11567→0.08340. However, region-equal paired improvement occurred in only 3/7 regions; the region-cluster bootstrap interval for ridge-minus-NNLS MAE was [−0.06765,+0.00825], crossing zero. Ridge worsened North in both years and Southwest in both years; it also slightly worsened East and South. NBS nested ridge beat the Ember baseline in pooled MAE (0.05144 vs 0.06951), but not significantly by seven-region resampling.

## Interpretation

This is a meaningful diagnostic and a promising robustness lever: weak regularization can suppress unstable high-intensity group coefficients and sharply reduce the Northwest/Northeast overprediction. It does not yet meet the frozen rule requiring improvement in at least four regions. The aggregate improvement is driven by correcting severe failures, while the same penalty degrades well-supported regions. Ordinary ridge regression is also not, by itself, a novel scientific contribution.

Next: test a pre-registered support-adaptive blend that applies the regularized model only when the unregularized coefficients or held-out feature composition indicate weak support, with all decisions estimated inside training-region folds. Compare against the best fixed ridge and raw NNLS. Do not tune a Northwest-specific switch from these results. Continue to call the target annual consumption-side MEE factor; there is no hourly CIF or coverage claim here.


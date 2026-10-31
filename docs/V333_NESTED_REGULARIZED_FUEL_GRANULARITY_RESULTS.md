# V333 results: regularization stabilizes detailed groups, but the coarse schema remains better

Run date: 2026-10-30. Frozen design: [V333 protocol](V333_NESTED_REGULARIZED_FUEL_GRANULARITY_PROTOCOL.md). Outputs: `results/v333-nested-regularized-fuel-granularity-r1/`.

## Results

Nested spatial ridge improved both Ember schemas relative to their unregularized fits. Pooled province-weighted MAE fell from 0.0801 to 0.0571 kgCO₂/kWh for full8 and from 0.0695 to 0.0490 for coarse5. Region-equal MAE was 0.0552 for full8-ridge and 0.0511 for coarse5-ridge. Full8-ridge improved over raw coarse5 NNLS in 4/7 regions and both target years; its region-equal paired difference was −0.0163 with a wide seven-region bootstrap interval [−0.0680,+0.0352].

The fair same-source, same-training-budget comparison remains full8-ridge vs coarse5-ridge. Coarse5-ridge had lower pooled MAE (0.0490 vs 0.0571) and region-equal MAE (0.0511 vs 0.0552). Full8-ridge improved only 3/7 regions; paired difference was +0.00403 (full8 worse), interval [−0.01794,+0.03591]. Its main remaining failure is North, where full8-ridge region MAE exceeded coarse5-ridge by 0.0926. Full8-ridge beat NBS5 raw NNLS in pooled MAE (0.0571 vs 0.0746) and 5/7 regions, but the two feature products differ and the seven-cluster interval includes zero.

## Decision

Regularization explains part of the raw eight-group failure, but detailed fuel groups still do not provide robust incremental value over a regularized five-group Ember model. Keep fine-grained fuel data available for sensitivity and risk-specific bounds; do not claim it improves annual cold-start transfer by itself. The strongest annual model in these experiments is coarse five-group ridge, an ordinary regularized composition baseline, not a novel CIF predictor.

Next, stop searching for a better regression on the same 30-province, two-year target. Use the now sharper information hypothesis—fuel type alone misses conversion efficiency and cross-region source allocation—to test the value of an independent locally observed heat-rate/thermal-efficiency field or paired inflow descriptor on CEMS/hourly labels. Keep that evidence separate from Chinese deployment unless the matching fields are actually available and source vintages are verified.


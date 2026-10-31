# V336 results: training-selected heat-rate shrinkage does not beat fixed full transfer

Formal run: `results/v336-nested-heatrate-shrinkage-r2/`. V336-r1 is retained as an invalid comparator: it reused the jointly selected lambda in fixed-gamma arms instead of tuning each arm independently. The r2 run gives each fixed-gamma diagnostic its own inner-CV lambda.

## Results

The inner procedure typically chose gamma 0.5–0.75, indicating the training folds prefer partial efficiency adjustment. However, outer held-out gains over a separately tuned mix-only model were small and inconsistent. For NBS, nested generation-rate adjustment reduced region-equal pooled MAE 0.05450→0.05103, but improved only 3/7 regions; the paired region bootstrap was [−0.01100,+0.00206]. Nested supply-rate adjustment reduced it to 0.04956, improved 5/7 regions, but the interval [−0.01132,+0.00015] still crossed zero. The nested models improved 2023 substantially but had little or mixed value in 2022.

For Ember, the nested generation-rate model lowered region-equal MAE 0.05112→0.04947, with both years slightly improved and 4/7 regions better, but the interval [−0.00431,+0.00091] crossed zero. The nested supply-rate version improved only 3/7 regions and also had a wide interval. In every comparison, fixed full-rate transfer had a lower pooled point error than the training-selected gamma model. The selected partial-transfer rule did not prevent the known regional negative-transfer patterns.

## Interpretation

The hypothesis that inner-CV can learn a safer heat-rate transfer strength is not supported at this sample size. This is not an algorithmic increment. The point estimate is compatible with a modest gain, but seven regions and two years are too little evidence to claim adaptive gating. Do not add a learned province-level gamma to the proposed method based on these folds.

The strongest specific observation remains V335's fixed, interpretable thermal-efficiency adjustment on the Ember source, itself a narrow result requiring a true publication-time-valid target and an independent dataset. See the r2 CSVs and frozen [protocol](V336_NESTED_HEATRATE_SHRINKAGE_PROTOCOL.md).

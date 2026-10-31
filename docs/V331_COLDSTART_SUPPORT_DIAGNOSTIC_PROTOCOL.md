# V331 protocol: training-only support diagnostics for regional transfer error

Registered before running on 2026-10-30. V330 found train-only tuned weak ridge improves pooled cold-start MAE but helps only three of seven regions, with especially large gains in Northwest and Northeast and losses in North/Southwest. Before constructing any adaptive gate, V331 tests whether simple available diagnostics identify when ridge helps.

## Frozen data and outer predictions

Reuse the V329/V330 seven whole-region outer splits and 2022/2023 forward targets. Reproduce NBS NNLS, nested ridge-NNLS, and Ember NNLS predictions exactly. For each held province, calculate three scores without using held-region labels:

1. nearest training-composition distance: minimum Hellinger distance `sqrt(0.5 * sum((sqrt(p)-sqrt(q))²))` between its lagged NBS five-group share vector and a training row's feature vector;
2. regularization adjustment magnitude: `abs(NBS_ridge_prediction - NBS_NNLS_prediction)`;
3. cross-product prediction disagreement: `abs(NBS_NNLS_prediction - Ember_NNLS_prediction)`.

For each score, compare its association with outer absolute NNLS error and with the per-case gain of ridge over NNLS (`abs_error_NNLS - abs_error_ridge`; positive means ridge helps). Report Spearman correlation, region-cluster bootstrap interval, and AUC for classifying positive ridge gain. Bootstrap the seven region clusters, retaining both target years and all provinces in sampled regions. These are diagnostics only; no thresholds or gates are fitted in V331.

## Scope and decision

If a score has weak or region-dominated association with ridge gain, do not build a province-level gate from it. Even a positive diagnostic is only motivation for a new nested-gate experiment. Targets remain annual consumption-side MEE factors; this cannot validate hourly CIF or interval coverage.

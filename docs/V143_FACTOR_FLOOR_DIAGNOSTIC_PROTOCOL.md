# V143 — Oracle Factor-Contract Floor versus Reducible Interval Excess

## Purpose

V142's LGEE failure has two different forms. Under the assumed EF box, τ=900's proxy label interval itself is broad, while τ=600 has identical all-low/all-high K counts but wide predicted intervals. Separate these causes before designing another predictor.

This is a posthoc diagnostic on already prepared source and development responses, not a new model or confirmation. It introduces no target features and performs no calibration or fitting.

## Frozen quantities

Reuse the complete monthly-only panels from V140 and eligible V141 additions: 43 region identities and 936 region-months, two thresholds. For each observed hourly allocation and assumed EF box, let the saved endpoint counts be [K_low,K_high]. Define oracle factor-contract width F=K_high−K_low. Any interval containing the entire factor-box response must have width at least F. This is an evaluation oracle, not a computable target input under monthly-only information, and not a measured confidence interval for true Chinese factors.

Join V141/V142 predictions by region/month/threshold without changing any endpoints. Classify missed endpoint intervals into width<F (too narrow to contain the required interval at any location) and width≥F (location/one-sided extent failure). For each prediction compute the oracle expansion [min(pred_low,K_low),max(pred_high,K_high)] and its width. Report oracle repaired excess above F and covered-row excess width; these are diagnostic quantities, never deployable repairs.

Save every joined row and source floor, hashes, per-region and region-equal summaries. Assert physical monthly width≥F, label join identity, nonnegative oracle repair excess, original endpoints unchanged, and that every width<F row is uncovered. Plot monthly coal share against F/T for both thresholds, highlighting LGEE, PACE and SC. Report the assumptions and distinguish an EF-driven floor from dispatch-transfer error. No mechanism is promoted or genuinely unopened data accessed by this diagnostic.

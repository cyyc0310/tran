# Leave-one-BA-out risk calibration evidence

- Frozen protocol: `docs/CROSSFIT_DOMAIN_RISK_CALIBRATION_PROTOCOL_20261101.md`.
- Runner: `scripts/run_crossfit_domain_risk_calibration.py`.
- Post-hoc factorial diagnostic: `scripts/diagnose_crossfit_bias_radius_factorial.py`.
- Run directory: `results/crossfit-domain-risk-calibration-r2/`.
- Main outputs: `source_calibration.json`, `source_residuals.csv.gz`, `US_to_GB_predictions.csv.gz`, `prediction_seal.json`, `target_metrics.csv`, `target_scoring_hashes.json`, and `bias_radius_factorial.csv` with audit JSON.
- The first run (`r1`) generated invalid NaN metrics because missing hourly labels were not filtered. The corrected `r2` filters missing labels for calibration and scoring; `r1` is retained and must not be cited as valid evidence.
- GB 2024 labels were viewed in earlier work. Prediction was sealed before this run read the label, but this is a development replay, not blind confirmation.
- Leave-one-BA-out residual calibration improves nominal coverage mainly through wider intervals. It does not beat fixed-EF recomposition on the reported interval score, and the HGB bias correction is sensitive to calibration design.

Key SHA-256 values:

- Prediction file: `e880630cffb80c81a8ef7a24ca91c967e360c02a35d7875e3bd66e241a5cdd10`.
- Factorial CSV: `c260fc56852d82dcce882530cd83c4687ff8593e67cae7898942be138dda20fd`.
- Source calibration: `e096a8a598d14f73d9ce4f55ac6c30cffee86e7d7307da0cf89b289bec1b336f`.
- Target metrics: `c9eb6807a2b822fc9e48bb9be399773680a6e160061b06cd893dee89109e0f65`.
- Runner: `b068a0d635b8ce84ed073313d14e977dfe7e466af9eeb67a1fd2ba865a09ea2c`.
- Diagnostic: `1635d0d2664ef0b06f01ba652c883221daea3ac93359fbac33b5dbb33ef34e44`.

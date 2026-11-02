# Strict-CN5 pseudo-target factor-response r1

Run from the repository root:

```bash
python scripts/run_cn5_pseudotarget_factor_response.py \
  --output docs/evidence/cn5-pseudotarget-factor-response-r1
```

The protocol and results are in `docs/CN5_PSEUDOTARGET_FACTOR_RESPONSE_PROTOCOL_20261101.md` and `docs/CN5_PSEUDOTARGET_FACTOR_RESPONSE_RESULTS_20261101.md`. This is development-only nine-country LOCO, not blind confirmation. Strict target inputs are five monthly generation groups and calendar features; fine-fuel hourly labels supervise source models and pseudo-target folds only. The response is a fixed-factor generation-side proxy, not measured CIF.

Artifacts include the startup hash manifest, per-country H1 alpha score tables, nested pseudo-country alpha selections, sealed 2024 held-out hourly predictions and calibration radii, monthly/country/macro metrics, and run status. Prediction files and radii were written before held-target labels were scored.

Post-run SHA-256 verification is recorded in `post_run_verification.json`: input mismatches **0**, code/protocol mismatches **0**, prediction mismatches **0**, calibration-radius mismatches **0**. There are 9 held-out targets and 79,052 sealed hours.

Macro metrics (lower is better except coverage): ratio recomposition MAE 62.16, shape MAE 17.85, coverage 90.3%, width 389.94, interval score 534.77; pseudo factor response MAE 64.33, shape MAE 17.84, coverage 88.3%, width 350.23, interval score 505.44. The candidate narrows intervals and improves the pooled interval score, but loses coverage and worsens point MAE; HU and CZ remain material failure domains. See the results report for the no-go decision.

# CN5 monthly anchored relative-shape evidence

Formal scoring run: `results/cn5-monthly-anchored-relative-shape-r2/`. The experiment fits eight-source strict-CN5 models for each held-out country, calibrates on other-source 2023H1 hours, seals all nine 2024 predictions, and only then constructs target proxy labels for scoring.

## Frozen design

- Input: Energy-Charts nine-country panel with strict `t−2…t−7` thermal/hydro/nuclear/wind/solar monthly shares and calendar features.
- Candidate: monthly mean `anchor700` multiplied by nonnegative source-recomposition relative shape, normalized against a complete expected 24-hour month profile.
- Comparators: `anchor700`, `ratio_recomposition`, prior additive `daily_anchor_shape`, current strong `fuelshare_history`, `factor_history`, and `direct_history`.
- Label: common fixed-factor hourly generation proxy. All nine countries had appeared in prior project development; this is a nested source pseudo-target experiment, not a blind country confirmation.

## Files

- `*_predictions.csv.gz`: one sealed 2024 target prediction file per held-out country.
- `*_source_radii.json`: source-only 90% conformal radii for each method and fold.
- `*_monthly_anchor_audit.json`: target-month anchor level, expected 24-hour denominator, realized support mean, and numerical invariants.
- `country_metrics.csv`, `monthly_metrics.csv`: per-country and per-month scores.
- `macro_metrics.csv`: country-equal summary.
- `prediction_seal.json`: per-fold predictions, calibration files, source sets, and SHA-256 hashes.
- `startup_hashes.json`, `input_audit.json`, `run_manifest.json`: frozen code/data/protocol hashes and support/status audit.

Reproduce from repository root:

```bash
python scripts/run_cn5_monthly_anchored_relative_shape.py \
  --output results/cn5-monthly-anchored-relative-shape-replay
```

The r1 and formal r2 runs have identical numeric predictions (maximum absolute difference `0.0`). The candidate preserves the expected full-month anchor to floating-point precision; incomplete target-hour support can make the mean over observed rows differ, which is explicitly reported rather than renormalized using target labels.

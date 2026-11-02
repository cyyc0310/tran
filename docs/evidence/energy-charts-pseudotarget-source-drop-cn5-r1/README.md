# Strict CN5 pseudo-target source-drop r1

Run from the repository root:

```bash
python scripts/run_energy_charts_pseudotarget_source_drop_cn5.py \
  --output results/energy-charts-pseudotarget-source-drop-cn5-r1
```

The target budget contains five monthly shares and calendar only. Source deletion uses paired pseudo-target 2023H1 MAE changes; 2023H2 leave-one-country-out residuals calibrate the radii. The point predictions and radii are sealed before target 2024 labels are scored.

Primary artifacts under `results/energy-charts-pseudotarget-source-drop-cn5-r1/`: `selection.json`, `macro_metrics.csv`, `country_metrics.csv`, `monthly_metrics.csv`, `prediction_seal.json`, and sealed target forecasts.

SHA-256:

| File | SHA-256 |
|---|---|
| `startup_hashes.json` | `cc8388a367e44d213f701abd50dca924b827b3527e8185594b1d5e6ed2034e50` |
| `input_audit.json` | `d9bff82e170abab186e1f6e87801c0e6f8bbf965fb03faa9588dcda2f8bd2e78` |
| `selection.json` | `ed56a4c94260e342e36f77a23a133fe12a29bd86a53d31e3f1a8d866f5c12d66` |
| `prediction_seal.json` | `e0ed56024b92b20fa903ac43ac901307ce8fd2c3fad2a5a67c74b1032aa5c46c` |
| `monthly_metrics.csv` | `fb9f3565c337a21e72904857f7cea27ecbaa76ceef0f815094149397e11676ea` |
| `country_metrics.csv` | `308ffaf2780b63b3498db21ea2cb431817037f1a214c9bb1bd65d38ffbbe9893` |
| `macro_metrics.csv` | `46df83181398cc7bdcce41f4e7df879e2139f836c1dcf62110933f4520f6f668` |
| `status.json` | `c0cc5f3aeb4b486beaf846663c821a06d5dde6b18c979d17df9cd247fe7fdd4a` |

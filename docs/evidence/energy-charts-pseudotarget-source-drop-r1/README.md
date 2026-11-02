# Energy-Charts pseudo-target source-drop r1 evidence

Run from repository root:

```bash
python scripts/run_energy_charts_pseudotarget_source_drop.py \
  --output results/energy-charts-pseudotarget-source-drop-r1
```

The source set is fixed at NL/ES/BE/AT/SE/CZ/SK/HU/SI. The script uses only source-country labels in 2022 for fitting, 2023H1 for pseudo-target source deletion, and 2023H2 leave-one-country-out predictions for interval calibration. All 2024 target predictions and radii are sealed before scoring. The artifact is development evidence on fixed-factor generation-side proxy labels.

Primary files:

- `results/energy-charts-pseudotarget-source-drop-r1/selection.json` — candidate deletion contributions and selected source by outer target.
- `results/energy-charts-pseudotarget-source-drop-r1/macro_metrics.csv` — equal-country, equal-month summary.
- `results/energy-charts-pseudotarget-source-drop-r1/country_metrics.csv` — country results.
- `results/energy-charts-pseudotarget-source-drop-r1/monthly_metrics.csv` — monthly scores.
- `results/energy-charts-pseudotarget-source-drop-r1/prediction_seal.json` — point prediction and calibration-radius file hashes.
- `results/energy-charts-pseudotarget-source-drop-r1/*_heldout_predictions.csv.gz` — sealed predictions.

SHA-256:

| File | SHA-256 |
|---|---|
| `startup_hashes.json` | `03811b6e58fca070e41b601ab55f5626494ae0dfe01569da6a82a5d454331be8` |
| `input_audit.json` | `9310e5703599f717cf3a5636ff67210ab2f27b81d3ebabdd2323e3df8cceb6fe` |
| `selection.json` | `1c87a8ffd9a3c25f3253d33b6c1e3d52b10c7e2b917d29e705039a9d08cf7a88` |
| `prediction_seal.json` | `863116ffb468caf9e56b0fe748992efea043eebc3b8a1cdf89ba7d8daad4d19a` |
| `monthly_metrics.csv` | `1a1a9453435b655314f990dccc7b5c9d61fc3eeb769617a8303586bfca21fccd` |
| `country_metrics.csv` | `89d3b1143c576c3560175d39e9b286daff0e3ec01654cd83c3f4e6312f8e76dc` |
| `macro_metrics.csv` | `6b5dc17f9fba0d3eaaddc01e2af20485b01972400cf73e5c4b949028c1c702e0` |
| `status.json` | `f63d323eaccbdb399f16cc13b4f25ddd1ac2a83f39b33f85ef1c435d2784569e` |

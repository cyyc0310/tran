# Strict CN5 target-conditioned pseudo-target source selection r1

Run from the repository root:

```bash
python scripts/run_energy_charts_target_conditioned_source_drop_cn5.py \
  --output results/energy-charts-pseudotarget-target-conditioned-source-drop-cn5-r1
```

The experiment uses strict five-category monthly shares and calendar features, source-year 2022 labels, 2023H1 pseudo-target source selection, 2023H2 cross-fitted calibration, and sealed 2024 held-country predictions. Selection is repeated by target month using only that month’s lagged public CN5 features. Each held-country prediction and calibration radius is hashed before scoring.

Interpretation limits: the response is a fixed-factor generation-side proxy from revised hourly series; monthly inputs are back-calculated lagged simulations, not historical as-issued vintages. All nine countries were previously viewed development domains, not blind confirmation regions.

Primary artifacts are in `results/energy-charts-pseudotarget-target-conditioned-source-drop-cn5-r1/`: `selection.json`, `prediction_seal.json`, `monthly_metrics.csv`, `country_metrics.csv`, `macro_metrics.csv`, `status.json`, and nine sealed prediction/radius pairs.

SHA-256:

| File | SHA-256 |
|---|---|
| Runner | `b4af95e64649f2c930f706286b1188c4dbbdd9ef4a7ea23d2cbc9dff47b57ca0` |
| Protocol | `7626f854a319cafe92ea1136483d3c766bbc8593934d9337274527c9f37781f2` |
| `startup_hashes.json` | `63cb8395af4508b8dbec84a2304fad7ebda37254b55850b81e84aa03343216b0` |
| `input_audit.json` | `d9bff82e170abab186e1f6e87801c0e6f8bbf965fb03faa9588dcda2f8bd2e78` |
| `selection.json` | `2d4a32be10cb987389df2db138821b8c756aef2e62bf475fe41b94ca14e305ed` |
| `prediction_seal.json` | `1e4df988d1757928eecd2cbcf2f87a3e4c31ad9d5ad65b3f498ce05d155d4976` |
| `monthly_metrics.csv` | `567eaae33b46ecb1df29e772d3533950918701cc877216e9f8362d72013a86f4` |
| `country_metrics.csv` | `beef74c05fc27dcabf1fc20fb3485dfa6bbd8362526b251314be3e337d83cc5f` |
| `macro_metrics.csv` | `316143f29d785376555a919f81431436712cf0aba44d0b70209b4933e645d2d5` |
| `status.json` | `eeaed24b2e3d2735d8fcf9a6dd6b1b54e39f776a4c0e3726f900b461c9ef11e3` |

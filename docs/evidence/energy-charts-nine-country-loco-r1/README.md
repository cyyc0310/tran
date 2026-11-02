# Energy-Charts nine-country LOCO r1 evidence

Run from repository root:

```bash
python scripts/run_energy_charts_nine_country_loco.py \
  --output results/energy-charts-nine-country-loco-r1
```

The runner seals one 2024 prediction and one source-calibration radius file per target country before opening any target labels for scoring. `startup_hashes.json` records protocol, code, input manifest, raw data, and previous five-country result hashes. `prediction_seal.json` records each point-prediction and interval-radius file hash. `input_audit.json` contains timezone, source resolution, mapped-category nulls, complete-hour counts, and known schema gaps. The results are development evidence and use fixed-factor generation-side proxy labels.

Primary artifacts:

- `results/energy-charts-nine-country-loco-r1/macro_metrics.csv` — equal-country, equal-month summary.
- `results/energy-charts-nine-country-loco-r1/country_metrics.csv` — target-country results.
- `results/energy-charts-nine-country-loco-r1/monthly_metrics.csv` — target-month rows.
- `results/energy-charts-nine-country-loco-r1/old_target_source_expansion_pair.csv` — same-hour comparison of five-source and nine-source models on the original five targets.
- `results/energy-charts-nine-country-loco-r1/*_heldout_predictions.csv.gz` — sealed point predictions.

SHA-256:

| File | SHA-256 |
|---|---|
| `startup_hashes.json` | `02169bf803ebf8736cb4691aba7180d3f790bf6327e651e6ca4143cfb113bcac` |
| `input_audit.json` | `9310e5703599f717cf3a5636ff67210ab2f27b81d3ebabdd2323e3df8cceb6fe` |
| `prediction_seal.json` | `a7dfb2211ecf6661d7ce8cde5322174253a963c116f70fa3a8ead63849c6b0a2` |
| `monthly_metrics.csv` | `d5bf6302cc59b016ff1bdec9f0678561f6a0d4d5eec3ed2c54295cfd4d4b2a6b` |
| `country_metrics.csv` | `ae8ab4bde239717bf3fe9b81152684335089fea4cacc79e27154e9364e69f43f` |
| `macro_metrics.csv` | `d25e58f0bf695e237768d483b90274b00ccb04c6d9cd374c920279926243a097` |
| `old_target_source_expansion_pair.csv` | `2dd85ec6585779fcf523dae163e25b8ddc837d5b890722a1eae4a28cde040a8b` |
| `status.json` | `6d2ca3dbf8e934219175944c7f483485fff2a2a0a95259f9674073535b280907` |

# EIA–NESO official intensity transfer, M versus M+C

Frozen protocol: [`OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER_PROTOCOL_20261101.md`](../../OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER_PROTOCOL_20261101.md). Outcome summary: [`OFFICIAL_CONSUMPTION_CAPACITY_TRANSFER_RESULTS_20261101.md`](../../OFFICIAL_CONSUMPTION_CAPACITY_TRANSFER_RESULTS_20261101.md).

## Reproduction

From repository root, with the listed local raw source data installed:

```bash
PYTHONPATH=src .venv-research-py312-20261101/bin/python scripts/run_official_intensity_transfer.py \
  --output results/official-intensity-transfer-capacity-r1
```

The runner prepares source labels from 2022–23 and features that exclude response columns. It exits if any 2024 target-label files exist before the run. After all eight forecast files are sealed, it extracts the 2024 EIA/NESO responses, hashes them, and scores.

## Evidence contents

- `results/preseal_target_label_audit.json`: confirms the target-label folder was empty at startup.
- `results/prediction_seal.json`: SHA-256 and row count for all eight forecast files.
- `results/target_scoring_hashes.json`: hashes for the target responses extracted after the seal.
- `results/monthly_metrics.csv`, `region_metrics.csv`, `country_metrics.csv`: all metrics, including negative directions and every method.
- `results/startup_hashes.json`, `input_audit.json`, calibration files, and `status.json`.
- `code/`: experiment runner, source/target label preparation, shared feature module, annual-capacity budget builder, protocols, and result summary.
- `input/`: acquisition/coverage metadata, derived panel manifest, and the annual coal/gas capacity budget. Large raw EIA workbooks and NESO source CSV remain in the local ignored data directory; their source identities and derived artifact hashes are recorded in the manifests.

The EIA 2024 label unit identity check has maximum absolute discrepancy below 0.001 g/kWh. NESO’s 17,568 half-hour 2024 values form 8,784 complete UTC hours. EIA predictions cover 42,530 possible region-hours; 42,462 have target labels and 68 are missing. The study evaluates revised institutional estimates, not direct measurements. M+C uses 2020 coal/gas nameplate-capacity features and remains limited by UK-for-GB scope and current-vintage versions documented in the capacity protocol.

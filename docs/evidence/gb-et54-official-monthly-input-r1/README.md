# GB official monthly input transfer evidence, R1

This evidence package compares 2024 current-revised DESNZ ET 5.4 monthly MPP composition with NESO GB hourly generation-derived `t−2` five-group shares, then evaluates both input arms on the same US→GB fixed-EF generation-proxy task.

## Primary artifacts

- `monthly_composition_comparison.csv`: 12 monthly CN5 source-share comparisons.
- `composition_alignment_summary.json`: aggregate source-alignment metrics.
- `method_metrics.csv`, `paired_input_arm_deltas.csv`: six frozen methods under both input arms.
- `*_predictions.csv` and `prediction_seal.json`: both full-year 8,784-hour predictions and SHA-256 seals.
- `target_timestamps.csv`, `neso_lagged_input_shares.csv`, `et54_lagged_input_shares.csv`: the exact legal target inputs consumed before prediction.
- `startup_hashes.json`: 14 source-input, protocol and code hashes; all match the saved files.
- `postscore_target_label_sha256.json`: the target hourly fuel-label file hash, recorded only after the two prediction files were sealed.

## Data source and vintage

Official workbook: [ET_5.4_OCT_26.xlsx](https://assets.publishing.service.gov.uk/media/6ae0cba87cf9b224890467e4/ET_5.4_OCT_26.xlsx), published 2026-10-29 on the [DESNZ Energy Trends electricity page](https://www.gov.uk/government/statistics/electricity-section-5-energy-trends). Workbook SHA-256: `e888876812a4a6bd408b86d40075f2f0e05c153c99b342f72dd8339f99bb4dcb`. Its 2024 rows are current revised values, not archived as-issued versions. ET 5.4 monthly data cover Major Power Producers, while target labels use GB NESO generation, so the source-alignment and scope limitation must be retained.

NESO fuel source SHA-256: `103bc41ec970c1d58dab912cd1fae9df27afa0b869ebdd8ddb4d5db7840a704f`. The prediction runner reads target timestamps and legal `t−2` monthly generation inputs before sealing predictions; it reads 2024 hourly fuel labels only after the prediction files are sealed.

See the [results report](../../GB_OFFICIAL_MONTHLY_INPUT_TRANSFER_RESULTS_20261101.md), [alignment protocol](../../GB_OFFICIAL_MONTHLY_INPUT_ALIGNMENT_PROTOCOL_20261101.md), and [transfer protocol](../../GB_OFFICIAL_MONTHLY_INPUT_TRANSFER_PROTOCOL_20261101.md).

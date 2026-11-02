# November 2026 candidate with 2025 public capacity

Status: sealed separately from the original 2020-capacity prediction; target response has not been read. Seven output tables reuse the original target timestamps and EIA-923 August 2026 input. They preserve all original M and 2020 M+C fields, then add a 2025-capacity coal/gas share anchor and separately refitted capacity Ridge/boosting outputs with GB 2023 source calibration.

The predictor budget uses EIA-860 final 2025 registered-BA nameplate capacity for the seven US regions and DUKES 2026 Table 5.7 UK aggregate capacity as the GB source proxy. The source and mapping audit is [`PUBLIC_CAPACITY_2025_ASOF_NOV2026_AUDIT.md`](../../PUBLIC_CAPACITY_2025_ASOF_NOV2026_AUDIT.md). The original 2020 M+C seal remains unchanged.

Independent validation checked 22 input/output hashes. Across all seven files, every M point prediction and source-calibrated value differs from the primary seal by at most 4.55e-13 due to CSV float reserialization. Refit capacity Ridge/boosting predictions differ from their 2020-vintage counterparts by at most 4.55e-13. This shows that a single GB source provides no observed cross-domain capacity variation for the learned heads; it is a support limitation, not evidence that annual capacity has no predictive value. The fixed coal/gas share anchor does vary across BA, but it has no target labels yet and must not be described as an accuracy improvement.

`prediction_seal.json` records each file hash. `gb_source_calibration.json` contains source-only calibration. `provenance.json` records code, data, base forecast and output hashes. The candidate can be scored only after November 2026 target responses are released, on the same hours and against the original M and 2020 M+C forecasts.

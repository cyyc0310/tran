# Generation pseudo-target expert transfer r3

Canonical run for [protocol](../../CROSS_COUNTRY_GENERATION_PSEUDOTARGET_EXPERTS_PROTOCOL_20261102.md) and [results](../../CROSS_COUNTRY_GENERATION_PSEUDOTARGET_EXPERTS_RESULTS_20261102.md).

- `sealed_predictions.csv.gz`: 8,760 hourly predictions, sealed before reading GB target labels.
- `prediction_seal.json`: prediction SHA-256 plus runner, protocol, EIA source inputs, and NESO feature input hashes.
- `source_pseudotarget_validation.csv`, `selection.json`: seven held-out EIA BA pseudo-targets and selected blend.
- `source_calibration.json`: BA-equal source 2024 bias and 90% absolute residual radius.
- `metrics.csv`: post-seal GB 2025 score; GB had already been used in prior development.
- `target_scoring_hash.json`, `status.json`: score input identity and development status.

Audit performed after execution: prediction SHA matches; all 17 runner/protocol/data hashes match. The initial r1 run is incomplete because its default target-label path was stale; r2 scored successfully but r3 is canonical after hardening path-based hash revalidation. Never use r1 or r2 as the canonical evidence.

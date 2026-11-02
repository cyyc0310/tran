# Ireland target preflight evidence

This is a prediction-free eligibility audit for the frozen CN5 fuel-share model. No model predictions, CIF proxy scores, or error metrics were produced.

- Raw source: Energy-Charts API v2 `public_power`, country `ie`, request window `2022-01-01` to `2025-01-01`.
- License: CC BY 4.0; attribution: `energy-charts.info`.
- Retrieval: 2026-11-01 12:15:31 UTC.
- Reported timezone/resolution/unit: `Europe/Dublin` / `PT30M` / `MW`.
- Raw response: [`ie_2022_2024.json`](../../../data/energy-charts-blind-target-audit-r1/ie_2022_2024.json), 21,502,359 bytes, SHA-256 `936dcf77689d1f761191733f410aeb3641bf9702eac83c08acbd72e0881c09e7`.
- Metadata summary: [`ie_metadata_preflight.json`](../../../data/energy-charts-blind-target-audit-r1/ie_metadata_preflight.json).
- Eligibility decision: stop before prediction because the response contains `fossil_peat` and unresolved `others`, which cannot be mapped to the frozen fine-fuel labels without changing the target definition. The target is half-hourly and would need a frozen aggregation contract if re-opened.

The country has appeared in separate daily Carbon Monitor work in this repository. A future hourly Energy-Charts study would therefore be a new data/label-track replication, not a project-wide never-seen-country test.

# NESO 2024 embedded wind/solar forecast vintage subset

This evidence package preserves a fixed-origin extract from the official NESO annual archive. The raw archive is not duplicated here; its public URL, byte count and SHA-256 are in `manifest.json`. The subset is one published forecast snapshot per UTC delivery date, selected exactly as specified in the manifest. Forecasts were matched to NESO actual embedded wind/solar generation by settlement date and period, then independently confirmed to align one-to-one with the project's 2024 NESO half-hour generation panel.

- `forecast_origin_features_2024.csv`: 17,376 2024 target half-hours across 362 UTC delivery dates; includes UTC target interval start, settlement key, issue timestamp, lead, and embedded wind/solar forecast MW. It contains no target outcomes.
- `manifest.json`: raw/derived hashes, selection contract, coverage and forecast-vs-outturn diagnostics.

This is a **data-availability and forecast-relevance audit**, not a CIF prediction score, model comparison, or evidence of transferable benefit. Forecasts concern embedded wind/solar, not total generation by those technologies.

# US→GB 2025 official generation-intensity transfer evidence

Frozen task protocol: [`CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_GENERATION_PROTOCOL_20261102.md`](../../CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_GENERATION_PROTOCOL_20261102.md)

Results report: [`CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_GENERATION_RESULTS_20261102.md`](../../CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_GENERATION_RESULTS_20261102.md)

## Target archive

NESO Carbon Intensity API `intensity.actual`, 2025-01-01T00:00Z through 2026-01-01T00:00Z.

- Retrieved at: 2026-11-01T21:46:23Z.
- Half-hour intervals: 17,520 expected; 17,520 returned; 0 missing timestamps; 0 missing actual values.
- CSV SHA-256: `c6bb4e11b293457232b757b5e149db0a45f731bd1d6db27a7747baea10e42f14`.
- API chunk request/response hashes are recorded in [`manifest.json`](../neso-national-intensity-2025-r2/manifest.json); target CSV: [`neso_national_intensity_history.csv`](../neso-national-intensity-2025-r2/neso_national_intensity_history.csv).
- This is a current historical API response, not an archived as-issued vintage.

The first acquisition attempt ended with a TLS EOF before receiving data. The API request was retried; the complete acquisition finished successfully. The bounded retry handler is in [`acquire_neso_national_actual_intensity.py`](../../../scripts/acquire_neso_national_actual_intensity.py).

## Forecast and scoring bundle

Run output: [`results/us-to-gb-2025-official-generation-r4`](../../../results/us-to-gb-2025-official-generation-r4/)

- Prediction file: `sealed_predictions.csv.gz`, 8,760 rows, forecast hash is in `prediction_seal.json`.
- Source calibration: `source_calibration.json`, only EIA 2024 source labels were used.
- Score files: `overall_metrics.csv`, `monthly_metrics.csv`.
- Input, code, protocol and source-data hashes: `prediction_seal.json`; target response hash and successful seal verification: `scoring_manifest.json`.
- `audit.json` records the CN5 t−2 target-feature contract, source/training years, 2025 target hours and limits.
- Target NESO label file was first read by the scoring stage after the full target forecast was written and sealed.

## Reproduction

```bash
python scripts/acquire_neso_national_actual_intensity.py \
  --start 2025-01-01T00:00Z --end 2026-01-01T00:00Z \
  --output-dir docs/evidence/neso-national-intensity-2025-r2

PYTHONPATH=src python scripts/run_us_to_gb_2025_official_generation.py \
  --output results/us-to-gb-2025-official-generation-r4 \
  --target-labels docs/evidence/neso-national-intensity-2025-r2/neso_national_intensity_history.csv
```

The original empty failed-attempt directory `neso-national-intensity-2025-r1` is intentionally not used by the scored run.

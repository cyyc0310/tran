# EIA-930 official hourly CO₂ intensity labels, 2025

Retrieved 2026-11-01 from the official EIA-930 `region_data/series_data` endpoint for seven US balancing authorities. `CO2.GER` is generated CO₂ emissions rate; `CO2.CER` is consumed CO₂ emissions rate. API values are in lb CO₂/kWh. The raw responses are retained under `raw/`; normalized long-form rows are in `eia930_official_co2_2025.csv`; request URLs, per-series missingness, retrieval timestamps and hashes are in `acquisition_manifest.json`.

This is the current published historical series as retrieved, not an as-issued 2025 vintage and not direct smokestack measurement. Generation intensity is an EIA estimate derived from EIA-930 net generation and annual emission factors. Consumption intensity additionally uses EIA's MRIO allocation of import/export emissions. EIA warns that interchange imbalances affect consumption-intensity estimates. See https://www.eia.gov/electricity/930-content/about.html.

Coverage in the retrieval (timestamp counts include both requested endpoints):

- The API returns 8,761 hourly-ending timestamps per BA for the inclusive request. Convert each timestamp to a valid-hour-start by subtracting one hour, then restrict to 2025.
- Generation: 8,759 nonmissing, in-year aligned values per BA; the final in-year hour-start label is missing. Do not impute it.
- Consumption: 1,094 missing values per BA within the 8,760 in-year aligned hours, starting 2025-11-05 09:00 UTC; the additional missing value in the inclusive response is the final out-of-year endpoint. Use January–October for a common seven-BA consumption scoring window.

No model predictions or scores are included in this data package. Do not treat it as evidence of cross-country transfer performance.

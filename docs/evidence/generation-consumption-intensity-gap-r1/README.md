# EIA generation vs. consumption intensity boundary evidence

This is a measurement decomposition of the 2024 EIA-930 official intensity panel across BPAT, CISO, ERCO, FPL, ISNE, MISO, and NYIS. It compares generation-based intensity with consumption-based intensity on rows where both are available. It does not evaluate a transfer model or validate Chinese predictions.

## Files

- `summary.json`: complete-row count, conversion check, aggregate gap quantiles, exact decomposition, and input SHA-256 values.
- `by_ba.csv`: per-BA support and intensity-gap summaries.

The source files are the seven `*_official_intensity_2022_2024.csv.gz` panels in `data/eia930-hourly-co2-r1/`. The calculation uses `1000 × CO2 emissions / MWh` for g/kWh and checks the result against the published lb/kWh intensity columns using 453.59237 g/lb. The identity is documented in the main label audit.

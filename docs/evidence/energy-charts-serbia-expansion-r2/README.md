# Energy-Charts Serbia source expansion evidence

Six-country leave-one-country-out development experiment testing whether Serbia adds useful high-thermal/no-nuclear source support.

- The five original Energy-Charts country targets are paired against the frozen five-country r3 baseline.
- Serbia is a new development target; its target-year labels are scored only after all six prediction files are sealed.
- `input/` contains the exact Serbia API response and acquisition manifest. Original five-country inputs are in the local Energy-Charts panel and hashed in `startup_hashes.json`.
- `*_heldout_predictions.csv.gz`, `prediction_seal.json`, and `country_metrics.csv` preserve hourly predictions, seals, and scores.
- `input_audit.json` records category mappings, missing values, retained hours, and feature-eligible years.
- `source_support/` compares the range of nine coarse input coordinates in the five versus six source pool against 31 Chinese provincial inputs.
- `provenance.json` records input, code, baseline, and output hashes.

All labels are Energy-Charts fixed-factor generation proxies and monthly inputs are revised hourly aggregates with a simulated lag. This is not an as-issued or measured-CIF validation.

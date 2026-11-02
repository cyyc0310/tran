# Energy-Charts nine-country capacity r1 evidence

Run from repository root:

```bash
python scripts/fetch_eurostat_capacity_expansion.py
python scripts/parse_eurostat_energy_capacity.py \
  --input-dir data/eurostat-capacity-energy-charts-r2 \
  --output data/eurostat-capacity-energy-charts-r2/capacity.csv
python scripts/merge_eurostat_capacity_panel.py \
  --base-dir data/eurostat-capacity-energy-charts-r1 \
  --expansion-dir data/eurostat-capacity-energy-charts-r2 \
  --output data/eurostat-capacity-energy-charts-r2/capacity-nine-country.csv
python scripts/run_energy_charts_nine_country_capacity.py \
  --capacity-csv data/eurostat-capacity-energy-charts-r2/capacity-nine-country.csv \
  --capacity-manifest data/eurostat-capacity-energy-charts-r2/capacity-nine-country.manifest.json \
  --output results/energy-charts-nine-country-capacity-r1
```

The nine-country capacity CSV combines the five previously acquired Eurostat country slices with the four new raw responses in `data/eurostat-capacity-energy-charts-r2/`. The new requests use the official Eurostat dissemination API for `nrg_inf_epc`, filtered to annual net maximum capacity, total operator, MW, `CF` and `TOTAL`, years 2020–2022. `acquisition_manifest.json` records the request, retrieval time, response headers, and raw SHA-256. Responses are current revised values, not historical release vintages.

The runner seals one 2024 prediction file and one source-calibration radius file per target before scoring labels. Its startup hashes cover protocol, data acquisition/parser/merge scripts, model modules, raw generation inputs, capacity input and manifests. `prediction_seal.json` hashes both prediction and interval-radius artifacts. `input_audit.json` retains the known generation schema gaps from the base nine-country experiment.

Primary artifacts:

- `results/energy-charts-nine-country-capacity-r1/macro_metrics.csv` — equal-country and equal-month scores.
- `results/energy-charts-nine-country-capacity-r1/country_metrics.csv` — target-country scores.
- `results/energy-charts-nine-country-capacity-r1/capacity_increment_pair.csv` — capacity treatment minus its paired model without capacity.
- `results/energy-charts-nine-country-capacity-r1/*_heldout_predictions.csv.gz` — sealed predictions.

SHA-256:

| File | SHA-256 |
|---|---|
| `data/eurostat-capacity-energy-charts-r2/acquisition_manifest.json` | `18a0b13b3649e2cc208d650e79493b139e0241c283ad0190a675abe7211bf45d` |
| `data/eurostat-capacity-energy-charts-r2/capacity-nine-country.csv` | `0f9ee8904143dacf6dc3ef6064f705e5049c68dc6302ebfc2869df5c0d73959f` |
| `data/eurostat-capacity-energy-charts-r2/capacity-nine-country.manifest.json` | `59d569a5ce552a0daee770158a6495ef46d97d870cb3f30d37c38133d37afc53` |
| `results/energy-charts-nine-country-capacity-r1/startup_hashes.json` | `de21280869a7363ee395ff1e3577d0c9265c67324f39ed540a193b4400ce17b5` |
| `results/energy-charts-nine-country-capacity-r1/input_audit.json` | `9310e5703599f717cf3a5636ff67210ab2f27b81d3ebabdd2323e3df8cceb6fe` |
| `results/energy-charts-nine-country-capacity-r1/prediction_seal.json` | `533903d482e69513e77186c7e12ec6a2305848e2c9ffea75863874a042ef3235` |
| `results/energy-charts-nine-country-capacity-r1/monthly_metrics.csv` | `6d69469fb8567f7e5234ecdab84031794cdd2ca8145bbc59baa669b9c0bc01e6` |
| `results/energy-charts-nine-country-capacity-r1/country_metrics.csv` | `07d3fefdb483fe6f3e8ce799dbf411aa316e799dc501d562dbb4e7b7addc9c65` |
| `results/energy-charts-nine-country-capacity-r1/macro_metrics.csv` | `61a683cbeb0f3e776e1dd15911d78e5cef50924d3fe5f2bd8aba59e16b42dfa1` |
| `results/energy-charts-nine-country-capacity-r1/capacity_increment_pair.csv` | `5c1e30965f03ed99f8c4522828c116f644a23c97e58eb3d5f9c1be7ca70a4756` |
| `results/energy-charts-nine-country-capacity-r1/status.json` | `dd5e5c86e7748fe20c0c3e61e7645141961ce5280e13e1005ac6b78c5e3449a5` |

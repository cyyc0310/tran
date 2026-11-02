# Energy-Charts five-country LOCO evidence

This evidence bundle contains the five-country baseline and annual broad thermal-capacity increment runs for 2022–2024.

- `results/`: sealed target predictions, source-calibrated monthly and country metrics, startup code/input hashes, and data-quality audit.
- `input/`: five original Eurostat JSON-stat responses, parsed capacity table, and raw/output SHA manifest. Eurostat API dataset: `nrg_inf_epc`, queried for annual total operator net capacity in MW.
- `code/`: exact runner and parser snapshot, shared research modules, and both frozen protocols.

Energy-Charts hourly raw responses are public but large (about 181 MB) and remain in the local ignored path `data/energy-charts-v2-panel-r1/`. Their SHA-256 values and retrieval metadata are in the run's `input_audit.json` and `startup_hashes.json`; the API collection endpoint is `https://api.energy-charts.info/`. Attribution required by the API license: `energy-charts.info` (CC BY 4.0).

The runner wrote all five target prediction files and their hashes before opening 2024 labels for scoring. The monthly coarse input was reconstructed from the current revised hourly series, so this is a lagged-input simulation rather than a historical as-issued replay. Eurostat capacity snapshots are latest revised responses, not historical release vintages. The outcome is a fixed-factor generation proxy, not measured CIF.

The first capacity round (`../energy-charts-loco-capacity-r1/`) contains HGB increment heads; r2 adds the preregistered standardized Ridge head. The original M baselines remain numerically identical across both rounds.

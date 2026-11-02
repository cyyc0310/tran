# Energy-Charts five-country LOCO evidence (reproducible final run)

This bundle preserves the five-country baseline and annual broad thermal-capacity increment comparison for 2022–2024.

- `results/`: sealed target predictions, source-calibrated metrics, code/input hashes, and data-quality audits.
- `input/`: five original Eurostat JSON-stat responses, parsed capacity table, Eurostat manifest, and Energy-Charts acquisition/audit manifests.
- `code/`: exact runner, parser, shared research modules, and frozen protocols.

The 181 MB Energy-Charts hourly responses remain in the local ignored path `data/energy-charts-v2-panel-r1/`; all five response SHA-256 values and acquisition metadata are in `results/startup_hashes.json` and `results/input_audit.json`. The API attribution is `energy-charts.info` (CC BY 4.0).

All 7 runner code/protocol hashes and 5 held-out prediction file hashes were rechecked against this snapshot. Target prediction files were sealed before 2024 labels were scored. Coarse monthly inputs were reconstructed from current revised hourly values, and Eurostat values are current revisions rather than historic vintages. Labels are fixed-factor generation proxies, not measured CIF.

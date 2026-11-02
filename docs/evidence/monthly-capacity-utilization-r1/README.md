# Monthly capacity-utilization transfer evidence

This development bundle evaluates static annual technology-capacity shares and lagged monthly generation divided by lagged annual capacity on five Energy-Charts countries.

- `results/`: sealed 2024 outer predictions, per-country/month metrics, input audit, manifests, and prediction hashes.
- `../../MONTHLY_CAPACITY_UTILIZATION_TRANSFER_PROTOCOL_20261101.md`: frozen experiment contract.
- `../../MONTHLY_CAPACITY_UTILIZATION_TRANSFER_RESULTS_20261101.md`: interpretation and paired comparisons.
- Runner: `scripts/run_energy_charts_loco_utilization.py`.

All five targets were already viewed development countries; this is not unseen-country confirmation. Source training used 2022 and source calibration used 2023H1. Monthly inputs are current revised hourly aggregates with lag simulated; Eurostat capacity values are current revisions with lag simulated. They are not as-issued historical vintages. The response label is a fixed-factor generation proxy, not measured CIF. The prediction files were sealed before target-year labels were loaded for scoring.

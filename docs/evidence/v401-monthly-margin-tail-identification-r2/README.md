# V401 machine evidence

This r2 evidence package contains the corrected 2024 EIA seven-BA information-identification experiment, including both hourly-total (M+T) and daily-thermal-margin (M+T+D) oracle bounds. It contains 231 region-month-threshold rows. `summary.json` records the runner and source-data SHA-256 values; the detail CSV hash is recorded there too.

- Protocol: [V401 protocol](../../V401_MONTHLY_MARGIN_TAIL_IDENTIFICATION_PROTOCOL.md)
- Results and limits: [V401 results](../../V401_MONTHLY_MARGIN_TAIL_IDENTIFICATION_RESULTS.md)
- Runner: [v401_monthly_margin_tail_identification.py](../../../scripts/v401_monthly_margin_tail_identification.py)

The extra inputs are retrospective oracles. The 100% feasibility inclusion is not predictive coverage.

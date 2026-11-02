# Annual fossil heat-rate information test evidence

- Official raw input: `input/DUKES_5.6_2026.xlsx` (SHA-256 `1335dce766c36a3fa4a27be2ad951292b994877998839a9fdb21670957d30d64`).
- EIA-923 source ZIPs are referenced from `eia923-monthly-pseudotarget-r1/input` (2020–2021) and `eia923-ba-mapping-r1` (2022); exact hashes are in the experiment startup manifest.
- Runner: `scripts/run_thermal_rate_official_increment.py`.
- Frozen protocol: `docs/THERMAL_HEATRATE_OFFICIAL_INCREMENT_PROTOCOL_20261101.md`.
- Main output: `results/thermal-rate-official-increment-r3/` (regional metrics, annual rates, seals, startup hashes, target label hashes).
- Post-hoc center/radius decomposition: `scripts/diagnose_thermal_rate_interval_decomposition.py` and `results/thermal-rate-official-increment-r3/interval_decomposition.csv`; target data were already viewed, so this is diagnostic only.
- Evaluated on previously viewed 2024 target labels after predictions were sealed; this is a development replay, not blind confirmation.

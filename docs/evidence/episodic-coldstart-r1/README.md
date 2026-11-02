# Episodic cold-start transfer evidence

This evidence bundle contains the frozen five-country China-style information-budget experiment.

- `results/`: five sealed 2024 country prediction files, country/month metrics, source-calibration radii, status and prediction seals.
- `results/startup_hashes.json`: hashes of inputs and code at run start; all were rechecked after model fitting.
- Protocol and result interpretation: `../../EPISODIC_COLDSTART_TRANSFER_PROTOCOL_20261101.md` and `../../EPISODIC_COLDSTART_RESULTS_20261101.md`.
- Runner: `../../../scripts/run_episodic_coldstart_transfer.py`.

The five Energy-Charts hourly responses are retained in the repository's local data directory and their SHA-256 values are checked against the acquisition manifest. Monthly inputs are reconstructed from revised hourly values and apply exact two-month lagging. Labels are fixed-factor generation-side proxies. The 2024 target labels are scored only after all prediction files are sealed.

Audit performed after the run: five prediction files total 43,918 rows; every prediction is finite; every file SHA matches `prediction_seal.json`; all startup input/code SHA values match current files; `git diff --check` passes.

# Czech expansion r3 evidence

This is the authoritative corrected execution for the Czech source expansion.

- Protocol: `docs/ENERGY_CHARTS_CZECH_EXPANSION_PROTOCOL_20261101.md`
- Runner: `scripts/run_energy_charts_czech_expansion.py`
- Raw CZ response and acquisition manifest are copied under `input/`; original five country responses are referenced by absolute paths and SHA-256 in `startup_hashes.json` and `provenance.json`.
- Six target prediction files are sealed in `prediction_seal.json` before target labels are scored. Per-country/month metrics and the paired original-five comparison are retained.
- `input_audit.json` records all mapped-series null counts, hourly resolution, complete month counts, and usable feature hours.
- `macro_metrics_original_five_targets.csv` is the country-equal macro of the five existing targets, with Czechia included among the five training sources.
- `provenance.json` records input, code, baseline and output hashes. The file does not self-hash.

The earlier `r2` run executed the same predictions, but its file named `macro_metrics_original_five_targets.csv` contained country-level rows. It is retained for audit; use only r3 for aggregated reporting. An initial setup run failed before model fitting and is not evidence.

Run command:

```bash
PYTHONPATH=src python scripts/run_energy_charts_czech_expansion.py \
  --output docs/evidence/energy-charts-czech-expansion-r3
```

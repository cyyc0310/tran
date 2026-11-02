# Czech expansion r4 evidence

This is the authoritative execution, run after freezing the final protocol title and corrected aggregation.

- Protocol: `docs/ENERGY_CHARTS_CZECH_EXPANSION_PROTOCOL_20261101.md`
- Runner: `scripts/run_energy_charts_czech_expansion.py`
- Raw Czech response and acquisition manifest are copied under `input/`; the original five source responses are referenced by path and SHA-256 in `startup_hashes.json` and `provenance.json`.
- Six 2024 target predictions are sealed in `prediction_seal.json` before target labels are scored. Country/month metrics and the paired original-five comparison are retained.
- `input_audit.json` records the source intervals, nulls for mapped series, complete month counts and usable feature hours.
- `macro_metrics_original_five_targets.csv` is the country-equal macro for the five existing targets when CZ is included in training.
- `provenance.json` records input, code, baseline and output hashes. It does not self-hash.

The earlier r2 aggregation file contained country-level rows rather than the macro mean and is retained with a reporting note. r3 used the corrected aggregation but preceded the final protocol-title update. Use r4 for reported results.

Run command:

```bash
PYTHONPATH=src python scripts/run_energy_charts_czech_expansion.py \
  --output docs/evidence/energy-charts-czech-expansion-r4
```

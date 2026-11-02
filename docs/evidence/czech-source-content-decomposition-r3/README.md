# Czech source-content decomposition r3 evidence

Authoritative run for the matched-count source content and calibration decomposition.

- Protocol: `docs/CZECH_SOURCE_CONTENT_DECOMPOSITION_PROTOCOL_20261101.md`
- Runner: `scripts/run_czech_source_content_decomposition.py`
- Inputs: five original Energy-Charts country series, Czech raw response and manifests; their SHA-256 values are in `provenance.json`.
- Designs three source-training pools (old4, old4+CZ, old4+PX) crossed with three calibration pools (old4, old4+CZ, old4+PX), for each of the five original held-out targets.
- PX samples are no-replacement matched-count rows from old-source training or calibration pools. `pseudo_source_assignments.csv.gz` records selected identity, period and seed.
- All 15 target/training-arm forecasts are sealed in `prediction_seal.json` before target labels are scored.
- `country_metrics.csv` contains per-country/month-weighted results for all 3×3 source pool combinations and existing methods.
- `baseline_reproduction.json` records one historical BE/direct_history mismatch; it is excluded from comparisons to the archived prediction, while same-run contrasts remain paired. See protocol and result report for details.
- Runtime versions are saved in `status.json`. Inputs, baseline files, code and outputs are SHA-256 checked in `provenance.json`.

Run command:

```bash
PYTHONPATH=src python scripts/run_czech_source_content_decomposition.py \
  --output docs/evidence/czech-source-content-decomposition-r3
```

# Czech source-content decomposition r4 evidence

Authoritative run using package versions within the repository's declared ranges.

- Protocol: `docs/CZECH_SOURCE_CONTENT_DECOMPOSITION_PROTOCOL_20261101.md`
- Runner: `scripts/run_czech_source_content_decomposition.py`
- Runtime: Python 3.12.13, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.8.0, SciPy 1.17.1.
- Inputs and frozen five-country baseline files are SHA-256 listed in `provenance.json`.
- Three training pools (old4, old4+CZ, old4+PX) are crossed with three calibration pools (old4, old4+CZ, old4+PX) for five held-out target countries.
- PX rows are a fixed-size no-replacement source-pool sample. `pseudo_source_assignments.csv.gz` records target, seed, period and sampled row identity.
- All 15 target/training-arm forecast files are sealed before target labels are scored.
- `baseline_reproduction.json` records 40/40 shared target/method point predictions reproduced within 1e-8 under the supported runtime.
- Input, code, baseline and output hashes all verify. The provenance file does not self-hash.

Run command:

```bash
PYTHONPATH=src /path/to/python-3.12 \
  scripts/run_czech_source_content_decomposition.py \
  --output docs/evidence/czech-source-content-decomposition-r4
```

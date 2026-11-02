# CZ/PX matched-source seed sensitivity r1

This evidence package repeats the frozen CZ versus matched-count PX source-content decomposition with five predeclared root seeds: 20261102 through 20261106. The original 20261101 run remains available as the r4 anchor.

- Protocol: `docs/CZECH_SOURCE_CONTENT_DECOMPOSITION_PROTOCOL_20261101.md`
- Runner: `scripts/run_czech_source_content_decomposition.py`
- Runtime: `/tmp/transcif-research-env-py312-20261101` (Python 3.12.13, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.8.0, SciPy 1.17.1)
- Each `seed-*` directory contains the 3×3 train/calibration arms, 15 sealed forecasts, matched PX row identities, baseline reproduction check, and SHA-256 provenance.
- Each seed reproduces all 40 shared historical point predictions within `1e-8`; all 46 input/code/baseline/output hashes per run verify.
- `seed_macro_metrics.csv` concatenates each run's macro metrics. It is a derived summary; per-run `provenance.json` files remain authoritative for run outputs.
- No source seed was selected by target performance. The original targets and fixed-factor generation proxy remain development evidence, not blind confirmation or measured CIF.

Primary report: `docs/CZECH_PSEUDO_SOURCE_SEED_SENSITIVITY_RESULTS_20261101.md`.

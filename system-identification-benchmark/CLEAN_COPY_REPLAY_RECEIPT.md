# Replay receipts

## Earlier full InTaSet + Nano-Drone replay

The earlier candidate was fully rerun on macOS arm64 with CPython 3.12.13 and the pinned dependency lock. It verified 16 raw inputs (the included InTaSet ZIP plus 15 Nano-Drone Git LFS CSVs from pinned author commit `2d921b57d166fe2debe08a5d39bd07297c5abc39`) and generated 22 output files. All 11 scoring CSVs matched the prior clean replay byte-for-byte. Across the 22 outputs, 15 were identical; the remaining differences were run metadata or logs. After that run, local paths were redacted from logs and output hashes refreshed. During release packaging, one non-scoring log line's trailing display padding was removed and its hash refreshed; the scoring CSVs were not changed.

The earlier full run does not establish cross-platform bitwise reproducibility, blind confirmation, an InTaSet author Neural ODE reproduction, hardware safety bounds, or a stable algorithm winner. Nano-Drone inputs and author materials are not included in this package, and their specific license scope has not been verified. Its run receipt is preserved under `docs/algorithmic-control-v1/results/reproduction_release_candidate_20261103/`.

## InTaSet core replay from the public package ZIP

The InTaSet-only runner was executed from a clean extraction of the public package ZIP with CPython 3.12.13 and seeds `20261102,20261103,20261104`. It completed in 85.7 seconds and produced 148 master-summary rows. The included InTaSet ZIP passed its recorded source SHA-256 `a0aac3b0985bec232afa8f9e95cd87612515904f5e574bc483777e4d532e8c85`. All eight CSV outputs were byte-identical to the archived InTaSet results. This runner uses the included InTaSet data and does not fetch Nano-Drone sources. The package README and release manifest were updated afterward to include this receipt; the benchmark scripts, inputs, and score files were not changed.

## Cascaded Tanks clean-copy replay

This replay was run from a clean copy of this public-review package, using only the included Cascaded Tanks archive and the dependencies already installed at the versions in the package lock. It was run after the replay CLI and result-verification CLI were added.

- Runtime: CPython 3.12.13, NumPy 2.4.5, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.8.0.
- Source: the included 7,520,592-byte archive passed ZIP CRC, record MD5 `f832776ed1efdca4572338f467b7baec`, and SHA-256 `eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`. The verified 4TU source license is CC BY-SA 4.0.
- Results: 66 metric rows, 67,296 within-record prediction blocks, and 6 target-prefix calibration receipts. Predictions were finite. The result verifier passed.
- Exact replay: the metrics, block-level scores, calibration receipts, and excitation diagnostics were each byte-identical to the previously archived files. Their SHA-256 values are recorded in `docs/algorithmic-control-v1/results/cascaded_tanks_sparse_input_v1/cascaded_tanks_metadata.json` and the release manifest.
- Limit: this dataset has one estimation record and one public test record; the test output was inspected during source admission. This is a reproducible development result, not blind or independent confirmation. It uses recorded future pump inputs and block-start resets, not online control or full-run free rollout.

Commands used:

```bash
python scripts/verify_cascaded_tanks_source_v1.py
python scripts/cascaded_tanks_sparse_input_benchmark_v1.py \
  --output-dir /tmp/cascaded_tanks_replay
python scripts/verify_cascaded_tanks_sparse_input_benchmark_v1.py \
  --output-dir /tmp/cascaded_tanks_replay
```

# Cascaded Tanks clean-copy replay receipt

This replay was run from a clean copy of the public-review package, using the included Cascaded Tanks archive and the installed versions in the package dependency lock.

- Runtime: CPython 3.12.13, NumPy 2.4.5, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.8.0.
- Source verification: 7,520,592-byte archive; ZIP CRC passed; 9 archive members checked; 3 key member hashes checked; CSV/MAT arrays checked at 1,024 samples and 4-second sampling.
- Recorded source license: CC BY-SA 4.0. DOI: [10.4121/12960104](https://doi.org/10.4121/12960104).
- Fresh run: 66 metric rows, 67,296 prediction blocks, and 6 target-prefix calibration receipts. Result verifier status: `VERIFIED_CASCADED_TANKS_SPARSE_INPUT_RESULTS`.
- The four scoring/diagnostic outputs were byte-identical to the archived files:

| Output | SHA-256 |
|---|---|
| `cascaded_tanks_metrics.csv` | `95a8500c2abbe454513854b8bc00b4e7e2ba3f30735d3d5166b0ad7556cf48d0` |
| `cascaded_tanks_blocks.csv` | `3c4e14cb3c756db1d5718e39000e24a303ab0589a0bd15c19dd7c166c94f401d` |
| `cascaded_tanks_gamma_receipts.csv` | `651279bbf36856aa42e14330c2d69e5d7793f8c2f8fc52269d1842cbe76e362d` |
| `cascaded_tanks_excitation_diagnostics.json` | `89db96c764e000a0a3614404306de6c771c370c22a23830afb181faf15c2ad67` |

This proves reproducibility of the released computation on this host and input. It does not make the public target record a blind test: its labels were inspected during source admission, and blocks from one record are not independent physical repetitions. The task is offline conditional block prediction using recorded future pump input, not online or closed-loop control. The observed `y=10` saturation is not an independently verified physical safety limit.

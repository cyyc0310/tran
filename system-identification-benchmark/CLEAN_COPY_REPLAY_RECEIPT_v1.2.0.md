# Clean-copy replay receipt v1.2.0

Date: 2026-11-03. This receipt records verification from a newly extracted v1.2.0 release-candidate archive, separate from the research worktree. It verifies the included sources and archived outputs; it does not make the already inspected target record blind or provide another physical repeat.

## Environment

- CPython 3.12.13
- NumPy 2.4.5, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.8.0, PyTorch 2.12.0
- The main benchmark dependency lock was used.

## Verified from the clean extraction

| Verifier | Result |
|---|---|
| `verify_cascaded_tanks_source_v1.py` | Passed. Nine ZIP members checked; the source archive SHA-256 is `eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`; four CSV/MAT arrays matched within the recorded CSV precision. |
| `verify_cascaded_tanks_sparse_input_benchmark_v1.py` | Passed. 66 metrics, 67,296 prediction-block rows, and six target-calibration receipts were covered. |
| `verify_cascaded_tanks_arx2_regularized_ablation_v1.py` | Passed. Eight source-validation candidates; selected ridge `alpha=0.1`; 3,066 target predictions replayed; three horizon metrics recomputed. |
| `verify_cascaded_tanks_python_n4sid_sparse_benchmark_v1.py` | Passed. Eight source candidates refit; selected three Hankel block rows; 3,066 target predictions replayed with an independent Kalman-equation implementation. |
| `verify_cascaded_tanks_author_hybrid_sparse_benchmark_v1.py` | Passed. Three archived model seeds and 12 calibration receipts checked; 73,428 predictions replayed without retraining; 72 per-seed metrics and 24 summaries checked. |

The final archive manifest was checked from the extracted copy by recomputing every listed file size and SHA-256. The manifest intentionally excludes its own self-hash. InTaSet's clean-copy score replay from v1.1.0 is preserved unchanged; v1.2.0 does not modify its scripts, inputs, or archived score CSVs.

## Interpretation limits

The new hybrid and linear-model results use one public Cascaded Tanks test record whose labels were inspected during development. The three neural seeds are optimizer initializations, not independent physical trials. The recorded future pump input is provided within each offline prediction block. Observed output ranges are diagnostics, not hardware safety limits. The published hybrid is an adaptation of the Forgione–Piga model structure and main training recipe to this sparse-output protocol, not a reproduction of the original paper's score.

# Clean-copy replay receipt

Status: **the earlier local clean-copy replay passed; this public-review candidate is not a tagged release**.

- Runtime: CPython 3.12.13 with the included pinned dependency lock.
- Raw inputs used in that replay: 16 verified files: the included InTaSet Zenodo archive plus 15 Nano-Drone Git LFS CSVs obtained from the pinned author commit `2d921b57d166fe2debe08a5d39bd07297c5abc39`.
- Reproduction: 148 InTaSet summary rows and 450 Nano-Drone scored rows; 22 output files have SHA-256 entries in the run manifest.
- Comparison: all 11 scoring CSVs matched the historical clean replay byte-for-byte. Across all 22 output files, 15 were byte-identical; the other 7 differed only in run metadata or logs.
- Privacy and log hygiene: workspace and virtual-environment paths were replaced in logs; one log line's trailing display padding was removed and the output hash refreshed. The scoring CSVs were not changed.
- Recheck: the earlier candidate passed `scripts/reproduce_control_benchmark_v1.py --verify-only` after fetching the pinned inputs.
- Recorded warnings: author StandardScaler checkpoints were serialized with scikit-learn 1.7.2 and loaded under 1.8.0; a separate 1.7.2 replay had identical output files. SINDYc reports multiplication overflow in its feature construction; affected predictions remain counted as numerical failures.

This receipt documents a replay on the earlier candidate copy on macOS arm64. The current public-review candidate changes release documentation, licensing metadata, the file manifest, and one non-scoring log's whitespace; it has **not** been rescored after those packaging changes. No Nano-Drone source CSVs, author code, or checkpoints are included here. This receipt does not establish cross-platform bitwise reproducibility, blind confirmation, an InTaSet author Neural ODE reproduction, hardware safety bounds, or a stable algorithm winner.

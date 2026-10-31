# Nested fossil-role correction across sealed V50 point baselines

Run date: 2026-10-27. Protocol: [V50_NESTED_FOSSIL_STRONG_BASELINES_PROTOCOL.md](V50_NESTED_FOSSIL_STRONG_BASELINES_PROTOCOL.md). The analysis opened the same 70 external-source prediction packages and ten label packages; all five point arms were read from those sealed packages. No target/unseen artifact was opened and no prediction model was refit.

## Source-macro factor-2 results

| Point arm | Correction | CRPS | Coverage | Worst source | Width | Interval score |
|---|---|---:|---:|---:|---:|---:|
| Scale-anchor | Raw | 125.14 | 88.50% | 54.84% | 527.98 | 712.16 |
| Scale-anchor | EF-only | 71.43 | 95.27% | 75.18% | 478.64 | 510.42 |
| Scale-anchor | Nested fossil | 66.62 | 94.87% | 85.47% | 481.31 | 517.98 |
| B0 SizeET AlignedKeep | Raw | 138.11 | 86.02% | 49.65% | 516.52 | 763.54 |
| B0 SizeET AlignedKeep | EF-only | 67.93 | 95.43% | 74.61% | 468.95 | 502.10 |
| B0 SizeET AlignedKeep | Nested fossil | 66.20 | 93.94% | 74.61% | 469.00 | 515.11 |
| B1 SizeCapRidge AlignedKeep | Raw | 112.54 | 91.92% | 66.69% | 509.96 | 632.55 |
| B1 SizeCapRidge AlignedKeep | EF-only | 70.94 | 89.70% | 43.44% | 473.43 | 569.97 |
| B1 SizeCapRidge AlignedKeep | Nested fossil | 70.91 | 88.83% | 43.44% | 473.45 | 574.26 |

The two ReplaceR arms had severely undercovered factor-2 intervals (raw coverage 49.9% for B0 and 55.0% for B1); inner selection chose λ=0 on all 70 outer folds for each. Across aligned arms, the EF-only correction substantially reduces CRPS and interval score but still leaves large worst-source coverage failures. The fossil increment improves CRPS only modestly over EF-only; nested selection does not improve interval score or width on any arm. By arm, nonzero λ was selected 35/70 times for B0 AlignedKeep, 3/70 for B1 AlignedKeep, and 48/70 for Scale-anchor.

## Interpretation

Monthly composition can correct systematic point-level bias beyond a raw anchor, including on stronger aligned point arms. The fossil-role term does not yet add the required coverage-preserving interval efficiency over EF-only, and the method remains region-sensitive. This is not a validated risk method or a paper breakthrough. The ten external sources are reused development regions.

The producer wrote its summary and all four CSVs, then exited with status 1 because its final console-print statement requested a stale summary key (`daily_source_bias_mae`, renamed by the per-arm summary). The serialized result files are complete, their hashes are recorded in `summary.json`, and the summary's script hash matches the exact producer file used. This post-write reporting error does not alter the saved metrics; it should be corrected before any future rerun.

## Artifacts

- Script: `scripts/probe_v50_nested_fossil_strong_baselines.py`
- Output: `results/disjoint-calibration-v50-nested-fossil-strong-baselines/`
- Summary, per-source rows, bias summary, and nested λ selections are hash-bound there.

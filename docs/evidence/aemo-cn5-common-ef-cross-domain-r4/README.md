# AEMO common fixed-EF label sensitivity — r4 (superseded)

Historical run: `results/aemo-cn5-common-ef-cross-domain-r4/`. The reported metrics are unchanged in r5, but r5 is authoritative because it preserves each hourly fuel group's generation and the fixed-EF numerator for direct audit.

The runner verified the original sealed US/GB→AEMO prediction hash before rebuilding target labels. The original prediction file is SHA-256 `bcc11f64680df9c122655f5d8c1ffe337baf6f9a22cd10fa85f9efae2bd9bbb1`; the 2023 AEMO snapshots are checked against the source seal in `startup_hashes.json`. The new target label has SHA-256 `4d6fec0f873f6db5ad1b8390ef5ed4e00c6e3852d732de1cd4947415cffe0925` and contains 43,800 rows (8,760 unique hourly labels per each of five regions).

Scoring uses 21,663 paired target region-hours from NSW1, QLD1 and TAS1 after the frozen CN5 support rule and ≥99% mapped-fuel support filter. The exact EF map, label provenance, sample counts and limitations are in `summary.json`; region/macro metrics are in `region_metrics.csv` and `macro_metrics.csv`. `paired_scoring_rows.csv` stores the paired labels and sealed forecasts for independent rescoring.

Key hashes:

| Artifact | SHA-256 |
|---|---|
| Protocol | `e4f75039b1306a645a2bc276cceff4391b8b58e4310ec8a89b15c7b821b3d80c` |
| Runner | `15e48eeb17360bdb7fb0ac11fd943c75c2edea49c99b533e135fe2838a350aee` |
| Target labels | `4d6fec0f873f6db5ad1b8390ef5ed4e00c6e3852d732de1cd4947415cffe0925` |
| Region metrics | `2fe203aa1e669db54da2f4d4a96c5a86be4e8f4d718a04cf9781d6aa72d8f683` |
| Macro metrics | `112982c118d2f0a218982460b66107810fd1699037244c4ca0e864783befc846` |
| Paired scoring rows | `755ff26841176e86086f801544d4a1eaa82246c5f48089f59ae2e020fd71d12a` |

Reproduce with:

```bash
python scripts/run_aemo_cn5_common_ef_cross_domain.py \
  --output results/aemo-cn5-common-ef-cross-domain-rN
```

The AEMO staging data are local and are not copied into the repository. r1 failed before scoring on a timestamp dtype mismatch; r2's all-year label file retained one adjacent-month boundary hour per region and is marked invalid; r3 failed before label writing while applying the boundary filter. r4 is authoritative. r2 and r4 scored rows were identical because the extra rows were outside the registered scoring window, but only r4 is reported.

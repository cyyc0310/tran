# AEMO common fixed-EF label sensitivity — r5

Authoritative run: `results/aemo-cn5-common-ef-cross-domain-r5/`.

The runner verified the original sealed US/GB→AEMO prediction hash before rebuilding target labels. The original prediction file is SHA-256 `bcc11f64680df9c122655f5d8c1ffe337baf6f9a22cd10fa85f9efae2bd9bbb1`; all AEMO inputs were checked against the original prediction seal in `startup_hashes.json`. The target label has SHA-256 `740ba98b51be60f95be50a28854396cc95fb22510fadf88d09a4b8bfa8f0e419` and contains 43,800 rows (8,760 unique hours in each of five regions).

The target label CSV saves every hourly generation group (`generation_{fuel}_mw`), the fixed-EF numerator, mapped generation denominator, total registered positive MW and mapping support. Independently recomputed label closure was exact to floating-point tolerance. Region scores independently recomputed from paired rows matched `region_metrics.csv` within 1.9e-12. The common sample contains 21,663 region-hours: NSW1 7,253, QLD1 7,076 and TAS1 7,334.

| Artifact | SHA-256 |
|---|---|
| Protocol | `e4f75039b1306a645a2bc276cceff4391b8b58e4310ec8a89b15c7b821b3d80c` |
| Results report | `fd34fc0085a485ba0594b5604d806b4314d02bb63829f605d40a6259705711c9` |
| Runner | `421ab74eb6c85bc09fbd5bcf7e9975dc7d5ad8f92943c7c830c2066ab9af9224` |
| Target labels | `740ba98b51be60f95be50a28854396cc95fb22510fadf88d09a4b8bfa8f0e419` |
| Region metrics | `2fe203aa1e669db54da2f4d4a96c5a86be4e8f4d718a04cf9781d6aa72d8f683` |
| Macro metrics | `112982c118d2f0a218982460b66107810fd1699037244c4ca0e864783befc846` |
| Paired scoring rows | `d8fe954d5c52e4fdf1300cfb69c938a8f36c810262bd2c362764cf9a57c47201` |

Reproduce with:

```bash
python scripts/run_aemo_cn5_common_ef_cross_domain.py \
  --output results/aemo-cn5-common-ef-cross-domain-rN
```

The AEMO staging data are local and are not copied into the repository. r1 failed before scoring on a timestamp dtype mismatch; r2's all-year label file retained one adjacent-month boundary hour per region and is invalid; r3 failed before label writing while applying the boundary filter; r4 corrected month boundaries and yielded the same scores as r5, but did not save individual fuel-group hourly amounts. r5 is authoritative.

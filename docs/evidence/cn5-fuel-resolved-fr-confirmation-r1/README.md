# France external confirmation evidence

Reproduce from repository root:

```bash
python scripts/run_cn5_fuel_resolved_fr_confirmation.py \
  --output results/cn5-fuel-resolved-fr-confirmation-r1
```

The frozen external target is France, 2024-03 through 2024-12 (7,344 hours). The target was not used for prior CIF scoring. Only lagged CN5 monthly generation shares and calendar entered prediction. Nine existing development countries supplied 2022 source training and 2023H2 country-held-out pseudo-target calibration. France predictions and radii were sealed before scoring; all source, target, protocol, and code hashes were rechecked before target labels were read for metrics. See [the frozen protocol](../../CN5_FUEL_RESOLVED_FR_CONFIRMATION_PROTOCOL_20261101.md) and [results](../../CN5_FUEL_RESOLVED_FR_CONFIRMATION_RESULTS_20261101.md).

This is a single-country external replication of a fixed-factor generation-side proxy, not official CIF or measured emissions. France lacks separate `others` and `other_renewables` API series; the frozen mapping assumes absent categories are zero. The month values were reconstructed from revised hourly data, not historical as-issued releases. The 100% empirical interval coverage is overcoverage on one country/year and provides no finite-sample country-level guarantee.

## SHA-256

| Artifact | SHA-256 |
|---|---|
| Confirmation runner | `3b3a21a032908efa73fe71294ddd25bff32fe8a011266b56154bf73271cda438` |
| Frozen protocol | `90f650c0f3ead52cf392e18d490bde92e66bb725de8cd90cd9143f562bca0c41` |
| Results report | `419c838e77af794e1b1f00cb6def9c14546373288f298fe56220deb40163471a` |
| Startup code/input hashes | `eabc2b501963c8da0c1586b9f8d4b9cd2e3c23cc4b3196ba8c7b75ae02aff3d2` |
| Preflight audit | `7c2d841483900e478392b9fa0904df9ebb63d4a7fa59b7616c9aed9550de3261` |
| Prediction seal | `1b5ee295faa02e5d6b6789e9ed2de20fbf1249e3d50962c9019fd926e067417e` |
| Sealed FR predictions | `0494a8a92dfb9d1ed4ed0d8b991c52dac9fc1627cb2e516632e9778a42f1a59c` |
| Source cross-fit radii | `2a855ffc30997df9d452b92701c2db1198a81b789a27c6d1f138d620fdfe1917` |
| Monthly metrics | `7b646953540440f341d02a04170cdb267e2c81a5339a76e5d00a41b40a75504a` |
| Month-equal metrics | `c4fe11606a898246746ea2d8a93c4cd78e1e1af51d27fa024924ff326ba4196e` |
| Hour-pooled metrics | `a84c41b7b489a37a80b74178649684401fe103b391215478d10ee0177145cbf6` |
| Status | `ee80d60ee3c4f3275703033ee54fcb67e6d53246222e7b665ee2dcfbaa8376a7` |

The prediction and radius hashes are also recorded in `results/cn5-fuel-resolved-fr-confirmation-r1/prediction_seal.json`. The complete input and dependency hash map is in `startup_hashes.json`.

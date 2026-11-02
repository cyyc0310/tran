# Strict CN5 latest-month Ridge LOCO — r1

Authoritative output: `results/cn5-latest-ridge-loco-r1/`.

Nine previously viewed Energy-Charts countries were evaluated leave-one-country-out. Each fold trained on the other eight countries in 2022, calibrated symmetric 90% residual radii on their 2023H1 pseudo-targets, and sealed the held-out 2024 predictions before target fuel labels were opened. The candidate is a weighted standardized `Ridge(alpha=10)` on `LATEST`; strong existing references include anchor, ratio and macro recomposition, direct HGB and thermal-factor HGB.

The candidate did not pass the frozen gate: macro MAE 85.89 g/kWh versus 62.16 for ratio recomposition and 73.23 for anchor700; it beat ratio recomposition in 4/9 countries and anchor700 in 3/9. In the lagged-thermal-share ≥0.5 stratum, represented only by CZ and NL, coverage was 29.3% and interval score 1484.68, versus 49.5% and 940.61 for ratio recomposition. See the results report for the interpretation and limitations.

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `5972ca372570abf2c397e43b025df968cb77e9a372f286e2e863f7de9120d949` |
| Results report | `bbeb4016ae356f63760914a3068da04509b206d037a8abc9c51d4089df3ef0c6` |
| Runner | `7b6d06101b09953dd1c96c21dd6fd7b228766d025fe6704e918f5ee24fca8087` |
| Startup hashes | `1c70e7c8c824848746154e6e0bebf15315b0da15115cd848a9caf2eb3bdb4675` |
| Prediction seal | `f0625b0378438f7cda4675b56ec36740b2e860f080a141c9eb013b04c976f79b` |
| Macro metrics | `28c41e1802f03baa76476efdb72a85dd4aed4c220c8ae119d8fbe6fbc610d210` |
| Country-strata metrics | `cb075f77cd0bb50db1909ee2f177434733953e5e6c608505af4951113391bad0` |
| Monthly metrics | `56810bd5896229662c2c8aaa2f1802ed5ece176b67e7731307950e2ea8be9c41` |

Reproduce with:

```bash
python scripts/run_cn5_latest_ridge_loco.py \
  --output results/cn5-latest-ridge-loco-rN
```

The output directory must not already exist. This is a development experiment on previously viewed countries, not an untouched-geography confirmation. Labels are a fixed-factor generation proxy and public monthly histories are reconstructed from revised data rather than vintage-as-issued records.

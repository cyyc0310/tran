# CN5 fuel-share head — Portugal external hourly-domain confirmation, r3

Authoritative machine output: `results/cn5-fuel-resolved-pt-confirmation-r3/`.

Portugal's 2024 Energy-Charts hourly response was not used in prior TransCIF hourly scoring. The country was selected by schema/history completeness, after earlier FR/PL method results were already known; treat this as a sequential external replication, not a pristine pre-registered one-shot confirmation. The frozen 9-country strict-CN5 2022 source model and source cross-fitted 2023H2 q90 radii were not changed.

All 8,784 2024 local hours and all 12 lagged monthly input histories met the frozen support threshold. Predictions, radii, inputs and code were sealed before the hourly target intensity was calculated. Independent input/code hash verification had no mismatches; prediction/radius seals verified; independent monthly MAE rescoring matched within `1.42e-14` g/kWh.

The fuel-share head's macro MAE was 33.40 g/kWh and Winkler score 326.07, versus 40.18 / 391.88 for ratio recomposition. Its 90% empirical coverage was 99.83% with width 325.44. Against the stronger scalar thermal-factor history head, fuel-share MAE was effectively tied (33.40 vs 33.44), with 7.9% narrower intervals and slightly lower Winkler (326.07 vs 353.62). These are very wide, over-covering source intervals, not evidence of calibrated 90% target coverage.

The target mapping follows `CN5_FUEL_RESOLVED_PT_CONFIRMATION_PROTOCOL_20261102.md`. ERSE reports the closure of Portugal's coal plants by 2021 and no operating nuclear power stations. The API `others` category is grouped into the project's EF=400 `other` bin; oil and other-renewables are not separately exposed, so this fixed-EF generation proxy has mapping limitations and is not official CIF or measured emissions.

The first two PT runner attempts (`r1`, `r2`) stopped before scoring while resolving the acquisition manifest and schema-zero categories; their output directories contain no results. An earlier Italy candidate also stopped before scoring because `other_renewables` was null over the full requested period; it was not imputed and no Italy prediction or score was generated. `r3` is the sole authoritative scored run.

## SHA-256

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `856e9ebcc99350bf0e88a90299bcb221c10e901b17c3fe6e8295aa1ed1324189` |
| Results report | `e3fdcc6a8fe1175722ebe7e7cb90e13703cf21203e1d2a111509da3bf02a6387` |
| Acquisition runner | `3a2a90b98a53c324f650c4dc7d8cb0c87b997c57213d43b1279529703a112703` |
| Forecast/scoring runner | `cf5f8c0358c0075858bd5a97d340041cc058cb93a797b0e1f6d4690308ce9c74` |
| Acquisition manifest | `7df93ffaca082af973533221c88e23bc6fbdf3b4d27a95f5aa59241dc981e7d2` |
| Raw Energy-Charts response | `f778f75e2c0d2c8336cfd5c9b6712c6724df14ba71257ad7d9251fde790fd069` |
| Prediction seal | `98264bebdc99bde87e701ea11aea27abecb367f7093dbc0750248b2e2947a952` |
| Sealed predictions | `8e160896fb045fd692a2a1f40ac817bfcdfccb982a00ef048e295a4e655a07fc` |
| Source cross-fit radii | `2a855ffc30997df9d452b92701c2db1198a81b789a27c6d1f138d620fdfe1917` |
| Monthly metrics | `ee4378511e9e8755deb970ba81e84ec4fa88f0d464d71face64531e359b84791` |
| Month-macro metrics | `b0599baa0b3ecb1baef3e21537a0118e902c3ba5b1bb4f35727ecdac856df8d4` |
| Seasonal metrics | `018ca2131b99c058f8c1882f9cb448f9c62cd6a38847966a1ed2ef6d8014723b` |
| Hour-pooled metrics | `fd574cf5c0005b1fb68a60162f21daf6ed7ce0860f012e787f6f26733f81d3da` |

Reproduce acquisition with `python scripts/acquire_energy_charts_pt_cn5_replication.py`, then run with:

```bash
python scripts/run_cn5_fuel_resolved_pt_confirmation.py \
  --output results/cn5-fuel-resolved-pt-confirmation-rN
```

The target data are provided by [Energy-Charts](https://api.energy-charts.info/) under CC BY 4.0; cite energy-charts.info.

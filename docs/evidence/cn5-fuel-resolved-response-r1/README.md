# CN5 fuel-resolved response r1 evidence

Run from the repository root:

```bash
python scripts/run_cn5_fuel_resolved_response.py \
  --output results/cn5-fuel-resolved-response-r1
```

This is a nine-country development LOCO using 2022 source training, 2023H1 pooled source calibration, and sealed 2024 hourly point predictions. The target budget is lagged CN5 (thermal, hydro, nuclear, wind and solar) plus calendar. Four source-only HistGradientBoosting heads predict thermal fuel shares; nonnegative outputs are projected to the simplex and translated through fixed fuel factors. All target labels were sealed before scoring in r1. The response is a fixed-factor generation-side proxy, not an official or measured CIF. Monthly inputs are back-calculated from revised hourly series, not historical as-issued observations. This is not blind confirmation.

The post-hoc H2 calibration sensitivity was run separately:

```bash
python scripts/analyze_cn5_fuel_resolved_h2_calibration.py \
  --output results/cn5-fuel-resolved-h2-calibration-r1
```

It leaves r1 points unchanged and reconstructs source-country cross-fitted 2023H2 calibration radii. Since the 2024 labels were opened to score r1, H2 calibration results are post-hoc development diagnostics and cannot be called independent confirmation.

The post-hoc composition mechanism diagnostic was run separately:

```bash
python scripts/analyze_cn5_fuel_resolved_mechanism.py \
  --output results/cn5-fuel-resolved-mechanism-r1
```

It compares the source-trained dynamic share predictions with the source-training fixed mean shares on the same 2024 target hours. The labels had already been scored in r1; this is explanation of the observed gain, not another confirmation experiment.

## SHA-256

| Artifact | SHA-256 |
|---|---|
| Fuel-share runner | `02577c77d647d6c55d01c8cd092ba8bb68ac1c3896ef9934c5c576c962347047` |
| H2 analysis script | `fae4bbfd15b785cb8b1810ea099a76df30e7f89da254e52a7a3cb403f0ac9daa` |
| Fuel-share protocol | `34d3128e5b09ffcb03da4aa84a85862ada0bbe026681cdefdf7c97c3e52d83d7` |
| H2 protocol | `ae00f338554f98ce2b0a460f35d55ab8e386ab2fc73d4be2cb09a3c846c0f0c4` |
| Results report | `e7cf256b1068fba08142aef657a6db46122477ed489b30a2e3dfc450af550019` |
| r1 startup hashes | `3cff533a8d33cbd96b7db228fbbcbcf1e1dbd86f7452d6df764db29495985ac4` |
| r1 input audit | `d9bff82e170abab186e1f6e87801c0e6f8bbf965fb03faa9588dcda2f8bd2e78` |
| r1 prediction seal | `adacd8238330f7f2f0a8fd6cf4d610d56537f4a789e2aa3e8d357eb0d4e9c9e5` |
| r1 monthly metrics | `884c2546ab0d7355890296d3a50feff5356113d2b3ab74392ad10cec3ec0adfa` |
| r1 country metrics | `6ee6d574bc6d9baf72f2d74995d6ce63a01be39ab809230947a827917017207e` |
| r1 macro metrics | `0646b787471ef26a519fbe001820a81f72f206b1fbcc81b0188997bfa2b44630` |
| r1 status | `a16f65b7a034691b2051a070625bf65702ac2ea57b1d1178e782e29c3fdc9815` |
| H2 startup hashes | `5dbda7618dac0fc9411cbedd765c6dea7233cbd2629c24f462a71705becdb916` |
| H2 cross-fit radii | `7b712c6362e603133641aa41a7b91d7c616f4480ce41c563528f01b1d5b6fba7` |
| H2 monthly metrics | `1f612bd726b00fd45d3ca5b8ea41c714b817e12d5181acf1a1cf860847483600` |
| H2 country metrics | `19b19a780d5f0b1e45ba3b80bf64538e7f5c74135b6acda409e7602d2c830e90` |
| H2 macro metrics | `e243df8222f9264b19c213412d7005ad1da4db98c2ad6495f1a0f76eb0dd2457` |
| H2 status | `bc5082cbf212a6868cdcd1aa381aeb260ee645c3619ad48cfad7236e27a512b0` |
| Mechanism diagnostic script | `05923c0e7df37a2bc68ccd15ce79ac0f98fec21ff3ccc54a7f8f4322ca39302d` |
| Mechanism summary metrics | `1181fcac92fb4250ff84f3a803a6432d91335ede7dc1bfd48ced70dec6d9a02a` |
| Mechanism hourly records | `bc775bb8bd0412a36161e32083e532c6988f7ba6d498b75d961377739d4d6f91` |
| Mechanism status | `1afd1f573f9b3e8627120883d3614e10dd4d5d62e0c9293ee7d1f1351b5232a9` |

Prediction and source-radius hashes for all nine targets are listed in `results/cn5-fuel-resolved-response-r1/prediction_seal.json`. Full input/code hashes are in each run's `startup_hashes.json`.

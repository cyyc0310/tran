# CN5 softmax composition transfer r1 evidence

Protocol: `docs/CN5_SOFTMAX_COMPOSITION_TRANSFER_PROTOCOL_20261101.md`  
Result: `docs/CN5_SOFTMAX_COMPOSITION_TRANSFER_RESULTS_20261101.md`  
Runner: `scripts/run_cn5_softmax_composition_transfer.py`  
Raw outputs: `results/cn5-softmax-composition-r1/`

The runner records source input/code hashes in `startup_hashes.json`, stores all nine outer-target 2024 predictions and H2 source calibration radii before target scoring, and records their SHA-256 values in `prediction_seal.json`. It checks startup inputs/code and sealed prediction/radius files before scoring.

## Key artifact hashes

SHA-256 values below are from the completed run and tracked working tree at report time.

| Artifact | SHA-256 |
|---|---|
| Runner | `5ba81408024d74e688926c930ca335745eb46540f724839bc2b7ec9aa2658cad` |
| Protocol | `5dd3097fbbe4660b70acac01d9621cde9b8f4c2ca9bddbd3277012700fb9c166` |
| Startup input/code manifest | `1302b1a4a51b706216bb58099941f1a9550e3a96a34ed6ee663c0a18877fae2c` |
| Nested selection | `ad8cd494a688c9db35acd3a15ec4ceba60511652628d6f38e2fb6e035c63f9dd` |
| Prediction and radius seal | `72969d08f4da0b8aeb55d5adf37ee9931ee17bf89377706688be9549e9c86b93` |
| Inner pseudo-target scores | `bbfdf2e50bb5e6ff9ff8d316b667bc94678bbacfc6f8094c930fef40129ed50` |
| Country metrics | `137dc72a19c1fdc8b83b86f66f1d34977f42ff722b712e354eada931933d7c73` |
| Macro metrics | `46e015913349111bc93ce4764f9f74585a68a1691e21aeebd4e3420d6b0a9b72` |
| Fuel composition diagnostics | `54a57fddb0b4e72811855c9250cea91f87b364054c4734228ce520c2ed10dfa4` |
| Machine summary | `59dffb0a622ecb300b267850828d45e106ee47a6c3fae80f20b84c6a2a9239e1` |

The experiment is a previously viewed nine-country development replay, not a blind confirmation. Labels are fixed-factor generation-side proxies. See the result report for performance and limitations.

# Fine-fuel dual-track evidence (r3)

Frozen experiment: [protocol](../../FINE_FUEL_DUAL_TRACK_PROTOCOL_20261101.md); interpretation: [results](../../FINE_FUEL_DUAL_TRACK_RESULTS_20261101.md).

Final run is `results/fine-fuel-dual-track-r3/`. Source training labels are 2022, source calibration 2023, target development scoring 2024-02/04/06. Generation proxy and official consumption labels are trained, calibrated, and scored separately. Sixteen target prediction files were sealed before the scoring phase. The official consumption label coverage is incomplete for FPL/MISO in June; actual per-cell usable hour counts are in `region_month_metrics.csv`. All results are previously viewed development evidence; target M+F inputs are a fine-fuel upper budget, not China's five-group budget.

| File | Bytes | SHA-256 |
|---|---:|---|
| `GB_to_BPAT_consumption_official_predictions.csv.gz` | 21717 | `930584557cff3a0062bb332a73fdaee49366fe78c83df22c7eff2caa93263b2a` |
| `GB_to_BPAT_generation_proxy_predictions.csv.gz` | 21794 | `c77a0c9a22b6dfe945a05b49200ad2a1c3e8dbe74c5c641dfc4d927e94e37774` |
| `GB_to_CISO_consumption_official_predictions.csv.gz` | 26293 | `799742791d66da338586f7bc3687d335e6a2c4c47f831432193cc5346ce556e7` |
| `GB_to_CISO_generation_proxy_predictions.csv.gz` | 24873 | `57cb38733549c6604658580c2f7cc3dd35bdb25e3d1399486fe5cc4a76bf42d7` |
| `GB_to_ERCO_consumption_official_predictions.csv.gz` | 22164 | `bdcc6ea35faae4633486ae8c5168f814c92afd7818b86fdda54f58620c720586` |
| `GB_to_ERCO_generation_proxy_predictions.csv.gz` | 22098 | `c86d609a665f049572346070ff9f2d555bd87586dd5729937007214ae6872374` |
| `GB_to_FPL_consumption_official_predictions.csv.gz` | 26616 | `0463fd3f3d634ec76ed6de2f56d46f8d35d456a5eeb6c7cecf46049cff2d583e` |
| `GB_to_FPL_generation_proxy_predictions.csv.gz` | 26449 | `acbb57e920bb8ed804729e8354cc54c6839f0ccbfdca1b502f6fda87d78f7055` |
| `GB_to_ISNE_consumption_official_predictions.csv.gz` | 26631 | `d40f395202324a20252a24a844f67b97c073b81d09899c7aea19c5a53877008a` |
| `GB_to_ISNE_generation_proxy_predictions.csv.gz` | 26881 | `5d51b1d13fc0afee18c034d90000da7ef6b6a9b637b17a166bc227ca4b17d250` |
| `GB_to_MISO_consumption_official_predictions.csv.gz` | 26855 | `fbc09a524cb62d1d59d4b00fe5370a44613f724219a7628f5cfbbe964835ee23` |
| `GB_to_MISO_generation_proxy_predictions.csv.gz` | 26979 | `b272bd08f83cf8470f57dfc949e63ef199f7742a181acce145be4612efd97d39` |
| `GB_to_NYIS_consumption_official_predictions.csv.gz` | 23047 | `33d5ea9df1632d8583b312fe5b07727f2bedafbe797d0f95d99a116d005bff02` |
| `GB_to_NYIS_generation_proxy_predictions.csv.gz` | 23450 | `39014f7f4217d8011f728bd6cc4ed22f2556bbe2f355ff61178612cedc42dd2c` |
| `US_to_GB_consumption_official_predictions.csv.gz` | 26131 | `4b13a417d8c6420d360652513fc1eddc4d27d48311f0f68938adc53406262beb` |
| `US_to_GB_generation_proxy_predictions.csv.gz` | 26636 | `809167b68e2b1805b5b713742b1750ba278f422b7db5673a2629a4cb35ac432f` |
| `input_code_hashes.json` | 7554 | `717a11166debc26bee74b2b73cd975823c6ea0a0505dacb15773d8a622eb19a3` |
| `macro_metrics.csv` | 18515 | `667917ce5fb52f6c7a2a69592759047f619d322a4fc20103a46a31a64df1fe0d` |
| `prediction_seal.json` | 3975 | `bfd67aca3a7a74fd1b6e69216271cf3d41de80280f457eac516e38dd64f9b809` |
| `region_month_metrics.csv` | 58158 | `d1c4adab6bed962f40c9362aad0637a72bfa02536862b2b8674cc3f9e676fd46` |
| `source_calibration.json` | 10549 | `01c1e41002509e6ad1529d055f8ec497c9a85f636a90918888df514773fa750f` |
| `summary.json` | 8063 | `fccd530f9e8c4e9f716c8073b83ac77316b23fe47ff3ab7aa2dbe2768f6c0e7d` |
| `target_label_hashes.json` | 1549 | `f73cbb554737b9a11d767eb6b37669f57f2aa34b2f9962ff5718737232518c8c` |

# Strict CN5 log-ratio HGB transfer r1 evidence

Protocol: `docs/CN5_LOGRATIO_HGB_TRANSFER_PROTOCOL_20261101.md`  
Result: `docs/CN5_LOGRATIO_HGB_TRANSFER_RESULTS_20261101.md`  
Runner: `scripts/run_cn5_logratio_hgb_transfer.py`  
Raw output: `results/cn5-logratio-hgb-transfer-r1/`

The runner records SHA-256 hashes of inputs and code at startup, produces outer-target predictions and source calibration radii for all nine held-out countries, seals those 18 files, verifies hashes, and only then reads target fine-fuel labels for scoring.

## Key hashes

| Artifact | SHA-256 |
|---|---|
| Runner | `46cd4c8c659b9a0e1099449b8a44b9898698ca6bb41381bd08375886c2bb974e` |
| Protocol | `fdd113e709f8559150658279250832755ab131314409b26e1c8cda05dc464b46` |
| Startup input/code manifest | `b91932e8ae763ea08709317ff8395f6d3f89fe03085aee32addfdbed67d34afe` |
| Nested selection record | `67a59d6d38464089acf8c0d295c9f766b720b6d80eefe1e6697c953b787f3037` |
| Prediction/radius seal | `c24375e3c795f816631372b7ec61429ec1327f818db8f2003b32ba1d4b705ca2` |
| Monthly metrics | `15a451c6537f43eeb67a7cd07896c0927436f4d64c8e7c3e18803a3dd55631d0` |
| Country metrics | `70ad2938be2395c64f78d1f499a70a1cae933d99408727506584adc2e077378f` |
| Macro metrics | `08484b993b3eb6b4dc6d2e7c4b7ce117592240ff82be176885e2d936d3f2692a` |
| Fuel mechanism metrics | `f120014f4451dff2bac0f71e3c4020a0bbd19e3a81c41ada3f99d168b7743e53` |
| Machine summary | `8a99a77875387d7f47ccded7b14fd9584ce48bf06389ce27f52e55c8b8535fe2` |

All nine countries had been seen in previous development. This is not a blind confirmation; labels are fixed-factor generation-side proxies and the monthly inputs are reconstructed from revised hourly series rather than as-issued vintages.

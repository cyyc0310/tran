# Strict CN5 pseudo-target gate evidence

Run:

```bash
python scripts/run_pseudotarget_coarse_conditional_stack_cn5.py \
  --output results/pseudotarget-coarse-conditional-stack-cn5-r1
```

The run sealed 9/9 held-out country prediction files (79,052 hours) before target scoring. The exact per-target hour counts and SHA-256 seals are in `../../../results/pseudotarget-coarse-conditional-stack-cn5-r1/prediction_seal.json`; `../../../results/pseudotarget-coarse-conditional-stack-cn5-r1/startup_hashes.json` records 7 code/protocol hashes and 14 input hashes. All hashes were rechecked after scoring and matched. Detailed method and results: [protocol](../../PSEUDOTARGET_COARSE_CONDITIONAL_STACK_CN5_PROTOCOL_20261101.md), [report](../../PSEUDOTARGET_COARSE_CONDITIONAL_STACK_CN5_RESULTS_20261101.md).

`other_renewable` is excluded from every public predictor feature, including pseudo-target gate fitting. It remains in the fixed-factor proxy label denominator. Source-month values are reconstructed from revised hourly data with a t−2 lag; this is not an as-issued release replay. The 2023H2 source calibration is pooled across held-out-source pseudo-targets and is not a country-level coverage guarantee. All nine countries are development domains.

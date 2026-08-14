# Phase 5.2 — Significance Test Summary

## 1. Primary: pooled paired test across 145 (region × seed) pairs

Each pair contributes one aggregate MAE per method. `delta = MAE(B) - MAE(A)`; **positive delta means A is better** (lower MAE). Holm-Bonferroni correction applied across the family.

| Comparison | median ΔMAE | 95% bootstrap CI | Wilcoxon p | Holm p | reject@0.05 | win-rate A better |
|---|---:|---|---:|---:|:---:|---:|
| joint_trained - basismix_plus | +3.30 | [0.19, 4.67] | 1.01e-04 | 3.02e-04 | ✅ | 58.6% |
| joint_trained - causal_plus | +3.36 | [0.06, 4.83] | 6.89e-05 | 2.76e-04 | ✅ | 60.0% |
| joint_trained - equal_plus | +2.90 | [-0.17, 4.47] | 1.95e-04 | 3.90e-04 | ✅ | 55.2% |
| joint_trained - persistence | +6.92 | [5.54, 7.81] | 2.04e-16 | 1.22e-15 | ✅ | 79.3% |
| joint_trained - best_single | +2.55 | [-0.34, 3.87] | 6.41e-04 | 6.41e-04 | ✅ | 55.2% |
| basismix_plus - causal_plus | +0.05 | [0.04, 0.07] | 3.07e-12 | 1.54e-11 | ✅ | 77.9% |
| basismix_plus - equal_plus | -0.14 | [-0.19, -0.10] | 4.80e-18 | 3.36e-17 | ✅ | 13.8% |
| basismix_plus - persistence | +4.21 | [2.87, 5.57] | 1.59e-25 | 1.27e-24 | ✅ | 99.3% |

Effect sizes (Wilcoxon r, |Z|/√N):  joint_trained: 0.32, joint_trained: 0.33, joint_trained: 0.31, joint_trained: 0.68, joint_trained: 0.28

## 2. PatchTST-supervised (external reference, 41.47)

- joint_trained median MAE: **40.53**
- joint_trained beats 41.47 on **55.2%** of pairs
- one-sample Wilcoxon (joint − 41.47 < 0), p = 3.15e-01
- _PatchTST-supervised is a single external number (no per-pair data), so this is a one-sample test on joint_trained MAE vs the constant 41.47, not a paired comparison._

## 3. Robustness: per-region Diebold-Mariano (temporal)

Harvey-adjusted DM on per-hour errors (n=8 subset pairs). Joint vs BasisMix+: **2**/8 significant after Holm.

| pair | joint MAE < basismix+ MAE? | joint MAE < persistence? |
|---|:---:|:---:|
| NSW1_seed0 | ❌ | ✅ |
| QLD1_seed0 | ❌ | ✅ |
| SA1_seed0 | ❌ | ✅ |
| UK_02_South_Scotland_seed0 | ❌ | ✅ |
| UK_08_West_Midlands_seed0 | ❌ | ✅ |
| US_BPAT_seed0 | ❌ | ❌ |
| US_PJM_seed0 | ✅ | ✅ |
| VIC1_seed0 | ❌ | ❌ |

_Per-region Harvey-adjusted DM test on per-hour errors (n_eval*24, h=24). Positive DM stat => joint model has larger loss => baseline better; negative => joint better._

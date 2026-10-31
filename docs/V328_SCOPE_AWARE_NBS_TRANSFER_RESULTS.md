# V328 results: explicit reporting-scope remainder did not improve annual EF transfer

**Run date:** 2026-10-30  
**Protocol:** [V328 protocol](V328_SCOPE_AWARE_NBS_TRANSFER_PROTOCOL.md)  
**Reproduction:** `python scripts/v328_scope_aware_nbs_transfer.py`  
**Outputs:** `results/v328-scope-aware-nbs-transfer-r1/`

## Question and construction

V327 used monthly NBS categories normalized within the above-designated-size reporting scope. V320 shows those monthly totals are lower than same-source annual all-plant generation. V328 tests whether the scope gap itself helps transfer MEE annual factors. For each previous feature year, the five reported-category proportions are multiplied by the observed monthly/all-plant annual generation ratio; a sixth feature is the remaining share. NNLS learns a separate nonnegative factor for that unobserved-generation share. The six shares sum to one. All LOPO folds, years, targets, and baselines are identical to V327.

## Results

| Test target | Method | MAE | RMSE | P90 absolute error |
|---|---|---:|---:|---:|
| 2022 | NBS normalized five-group mix | **0.05700** | **0.07375** | **0.12072** |
| 2022 | NBS scope-aware six-group mix | 0.05794 | 0.07521 | 0.13197 |
| 2023 | NBS normalized five-group mix | **0.04646** | **0.06730** | **0.11769** |
| 2023 | NBS scope-aware six-group mix | 0.04658 | 0.06885 | 0.12598 |
| pooled 2022–2023 | NBS normalized five-group mix | **0.05173** | **0.07060** | **0.11972** |
| pooled 2022–2023 | NBS scope-aware six-group mix | 0.05226 | 0.07210 | 0.13113 |

The scope-aware model is slightly worse in MAE in both years, and has worse pooled RMSE and P90 absolute error. Its paired absolute-error difference versus V327 NBS-five is +0.00053 kgCO₂/kWh (scope-aware minus five-group; province-cluster bootstrap 95% interval −0.00432 to +0.00520); 14/30 provinces favor the scope-aware model. It misses the pre-registered advance rule. The explicit residual category is **not supported** as a level-transfer improvement under this common-coefficient NNLS specification.

## Interpretation

V320's statistical coverage gap is crucial when treating monthly generation as a hard supply cap in a physical feasible set, as V321 established. But the same gap did not improve this separate annual cross-province regression when represented by one pooled sixth-group factor. The scope gap remains a limitation to quantify, not a feature with demonstrated predictive value.

V327's positive NBS-versus-Ember point estimate remains the better of the tested NBS level models, but its paired uncertainty interval crosses zero. Neither experiment supports an algorithm breakthrough. The next evaluation should test stronger geographic cold-start splits rather than add more fuel-share coefficients: leave whole Chinese statistical regions out, retain identical legal lagged features, and compare NBS, Ember, and training-region mean. This directly tests the user's “previously unseen area” goal.

## Limits

The outcome is still the official annual consumption-accounting factor, not hourly generation CIF. The NBS monthly series is above-designated-size coverage and its historical as-of vintage is unknown. The MEE training labels are published with substantial delay, so this experiment is a retrospective cross-sectional spatial transfer study, not an annual forecast available in real time. There are only two forward target years and 30 provinces.

Scores, province predictions, learned coefficients, scope ratios, and hashes are in `results/v328-scope-aware-nbs-transfer-r1/`.

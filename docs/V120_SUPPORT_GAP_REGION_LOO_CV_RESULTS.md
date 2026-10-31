# V120 — Support-Gap Fallback Region-LOO Cross-Validation Results

> Correction from V139: the 589-month denominator below includes 76 monthly-energy-infeasible capacity contracts. The old solver could return singleton intervals for those inputs; 44 threshold rows exclude their robust proxy endpoints. This archived table is invalid as overall performance evidence. See [the correction scope](CAPACITY_CONTRACT_CORRECTION.md) and [the corrected full-denominator comparison](V139_PHYSICAL_RESERVE_CALIBRATION_RESULTS.md). The historical numbers below are retained for provenance.

## Exact outer-fold evaluation

We evaluated 25 held-out calibration-source regions across 589 feasible region-months per threshold. For every fold, the test region was removed from both the anchor-error calibration and the support-gap boundary pool. All interval endpoints and hourly strict-threshold proxy counts use exact rational arithmetic. The four predeclared support-tail levels were 5%, 10%, 20%, and 30%.

| Threshold | Support-tail | Coverage, region-equal | Coverage, pooled | Width, region-equal | Capacity-only width | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| τ=600 | 5% | 92.48% | 92.36% | 154.3 h | 206.6 h | 0.18% |
|  | 10% | 92.48% | 92.36% | 154.9 h | 206.6 h | 0.35% |
|  | 20% | 92.83% | 92.70% | 161.2 h | 206.6 h | 2.42% |
|  | 30% | 92.83% | 92.70% | 179.8 h | 206.6 h | 10.02% |
| τ=900 | 5% | 99.33% | 99.32% | 41.5 h | 133.0 h | 2.67% |
|  | 10% | **99.83%** | **99.83%** | **70.5 h** | 133.0 h | 7.17% |
|  | 20% | **99.83%** | **99.83%** | 76.1 h | 133.0 h | 8.23% |
|  | 30% | **99.83%** | **99.83%** | 94.2 h | 133.0 h | 12.26% |

At τ=900, support-aware fallback improves coverage over the plain anchor-capacity intersection (97.0%) while keeping intervals 47% narrower than the physical capacity set at the 10% support-tail setting. At τ=600, the plain intersection is 92.30% covered; the best predeclared fallback levels raise it only to 92.83%, with a smaller width advantage. The support rule therefore appears useful mainly for high-threshold tail risk and does not yet solve the τ=600 coverage problem.

## Interpretation and next gate

This is the first cross-validated candidate with a clear joint gain at τ=900: region-level cross-fitting, a source-only risk-support alarm, and exact physical fallback retain almost all observed proxy coverage while reducing the capacity-set width. It is not yet a paper-level breakthrough. The 25 folds are a small, related U.S. balancing-area sample; calibration and evaluation use the same historical years under region-level cross-validation; target-quarter Q3 results were already opened; and the data are fixed-factor proxy labels. The 5–30% frontier should remain visible; choosing one operating point now would require an explicit risk-coverage objective.

Freeze the method and its chosen operating point before the next untouched region-time cohort. The next evaluation must include nonzero τ=900 support and enough τ=600 risk months, and should compare the frozen fallback against monthly-only, capacity-only, plain intersection, and strong dynamic/probabilistic baselines. A Chinese CIF claim still requires same-boundary hourly generation and emissions labels.

## Reproduction

- Frozen protocol: `V120_SUPPORT_GAP_REGION_LOO_CV_PROTOCOL.md`.
- Exact-rational runner: `../scripts/v120_support_gap_region_loo_cv.py`.
- Final exact output: `../results/v120-support-gap-region-loo-cv-r2/`. The preliminary floating-point run is preserved separately as r1 and must not be used for claims.

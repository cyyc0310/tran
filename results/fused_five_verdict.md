# Task 4.2 — Drop-One Verdict

**VERDICT: ZS_PLUS_WRAPPER**

- Best-single (Causal-alone) MAE: **42.101**
- Drop-Hier BasisMix+ MAE median: **53.780**
- Equal+ MAE median: **53.625**
- BasisMix+ MAE median: **53.745**

## Independent signals

- **R3 diversity**: DIVERSITY_MIRAGE (drop-Hier 53.780 vs best-single 42.101)
- **R1 meta-value**: META_DEAD_WEIGHT (equal+ 53.625 vs BasisMix+ 53.745)

## Drop-each-direction medians (BasisMix+)

| drop | MAE median |
|------|-----------|
| rag | 53.743 |
| phys | 53.782 |
| causal | 53.790 |
| icl | 53.707 |
| hier | 53.780 ← weakest drop |

## Implications for paper headline (Task 5.3)

Fusion contribution is NOT real: drop-Hier does not beat the
best single direction (R3 confirmed), and equal+ ≥ BasisMix+
(R1 dead weight). The head's apparent gains come from ZS+
test-time calibration, not from intelligently combining
directions.

**Paper claim (revised)**: 'TransCIF-ZS+ test-time calibration
is the workhorse. Five direction priors provide ensembling
robustness via equal-weight averaging, but a learned
meta-learner on top of ZS+ does not improve over equal-weight
+ ZS+ at this evaluation scale.'

**Action**: Retain the 5 single-direction + ZS+ story as the
headline. Drop BasisMix from the contribution narrative, or
relegate it to an ablation showing the meta-learner is dead
weight (R3 confirmed).

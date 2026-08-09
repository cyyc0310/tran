# Task 4.1 — Drop-one ablation

**VERDICT (Task 4.2): ZS_PLUS_WRAPPER** (drop-Hier BasisMix+ median = 53.780 vs best-single Causal = 42.101; R1: META_DEAD_WEIGHT; R3: DIVERSITY_MIRAGE)

Verdict (Task 4.2) is computed in a separate step. This table
shows MAE when each direction is dropped from BasisMix.

| target | drop_rag | drop_phys | drop_causal | drop_icl | drop_hier |
|--------|----------|-----------|-------------|----------|-----------|
| QLD1 | 27.225 | 27.140 | 27.145 | 27.160 | 27.140 |
| NSW1 | 46.771 | 46.777 | 46.819 | 46.687 | 46.774 |
| VIC1 | 98.740 | 98.691 | 98.594 | 98.584 | 98.655 |
| SA1 | 60.716 | 60.788 | 60.761 | 60.728 | 60.786 |
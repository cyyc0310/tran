# V121 — Frozen Support-Gap Rule on NWMT 2026Q3 Results

> V139 correction notice: source support calibration included capacity-infeasible months. A saved audit now recalibrates the cutoff on input-feasible source months; V121's six target threshold endpoints remain unchanged and covered. This is a posthoc correction using saved target endpoints, not a new raw-data rerun or new blind confirmation. See [correction scope](CAPACITY_CONTRACT_CORRECTION.md).

## Cohort

NWMT was selected from the remaining V52 regions using completeness only: its 2026H1 complete-proxy coverage was 84.5%, but all three 2026Q3 months had 100% valid proxy hours. It was not in the 43-region V118 target roster. Its identity was present in the 2024–2025 calibration source pool, so this is a new target-quarter confirmation in a historically known region, not a wholly unseen BA identity. NWMT was excluded from both its V121 anchor-calibration fold and its support-boundary fold.

## Results

The frozen rule was capacity-only at τ=600 and, at τ=900, anchor∩capacity unless normalized support gap exceeded the q=20% source boundary. All 3 months were capacity-feasible. At τ=900 the factor-box proxy K was exactly zero in all three months; the frozen method produced [0,81], [0,81], and [0,79] hour intervals, all containing the factor-box endpoint. Mean width was 80.3 h versus 504.3 h for the capacity set, an 84.1% reduction. The gap stayed below the 0.809 source boundary, so no fallback was triggered. All methods abstained at 1%, 5%, and 10% tail rates.

At τ=600, the observed nominal proxy counts were 224, 461, and 328 hours, while the ±10% endpoint ranges were [65,393], [270,628], and [143,479]. The frozen capacity-only interval contained every range but was broad: mean width 735 h. All methods abstained at the three decision rates. This confirms the lack of τ=600 efficiency gain.

## Interpretation

The frozen high-threshold rule transfers to a quarter not scored in V118 and keeps a narrow, proxy-covering interval in a lower-H1-completeness region. This remains an easy τ=900 zero-tail case with only three months, and it yields no actionable risk certificate. It supports interval efficiency in one new time cohort, but does not resolve the key failure on nonzero high-threshold support or establish transfer to a new BA identity or China.

## Reproduction

- Frozen protocol: `V121_NWMT_Q3_FROZEN_SUPPORT_FALLBACK_PROTOCOL.md`.
- Runner: `../scripts/v121_nwmt_q3_frozen_support_fallback.py`.
- Rows and hashes: `../results/v121-nwmt-q3-frozen-support-fallback-r1/`.

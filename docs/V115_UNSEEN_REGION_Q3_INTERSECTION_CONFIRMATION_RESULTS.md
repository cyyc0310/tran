# V115 results: spatial confirmation of the anchor-capacity intersection

## Cohort

SPA, TAL, TEPC, and WALC were selected before scoring using the V52 quality-only census (2026H1 complete-proxy coverage ≥90%) and absence from prior region-result documents/scoring scripts. The EIA-930 Q3 file has hourly timestamps for all 12 candidate region-months. After fuel-proxy completeness checks, 11/12 months met the ≥80% rule; WALC September had 46.7% coverage and was reported unavailable. The other 11 months passed group-energy capacity compatibility.

## Results on 11 eligible region-months

All four intervals—the factor-robust monthly anchor, monthly-only robust identified set, capacity-constrained set, and their intersection—contained the low-to-high factor-box proxy K endpoints in 100% of eligible rows.

| Threshold | Anchor width | Monthly-only set | Capacity set | Anchor ∩ capacity |
|---|---:|---:|---:|---:|
| 600 gCO₂/kWh | 338.1 h | 183.8 h | 161.0 h | **85.3 h** |
| 900 gCO₂/kWh | 259.5 h | 183.8 h | 126.3 h | **65.5 h** |

At each threshold and each 1/5/10% tail target, the region-equal macro frontier gives 75% negative certificates and 25% abstentions for the monthly-only, capacity, and intersection methods; there are no positive certificates. The anchor by itself abstains everywhere. The intersection therefore retains the capacity method's selective decisions while roughly halving its mean width, and is narrower than all three component/baseline intervals on this cohort.

## Why the result is promising—and still preliminary

This is the first spatially unseen-region evidence for the **intersection mechanism**. It tests four codes not used in prior result documents/scoring scripts, with Q3 labels held back until the method and roster were frozen. The result is not just a lower width obtained by discarding the physical set: the intersection retained the capacity set's negative certificates and covered the complete EF-box proxy K range in all 11 eligible rows.

The composition was designed after observing V104 and V113 on different regions, so it is a new spatial confirmation, not a new time holdout. Most certificates come from the easy zero-support regions SPA/TAL/WALC. TEPC is the harder case: nominal τ=600 K values are 22–69 hours and the factor-box upper endpoints reach 232 hours; the intersection remains abstaining but narrows the capacity interval to the empirically calibrated anchor range. At τ=900 all proxy K values are zero. The sample is small and EIA-proxy based; the calibrated anchor component has no formal finite-sample coverage guarantee. This is a promising candidate method, not yet a general CIF breakthrough.

## Reproducibility

- Frozen protocol: `docs/V115_UNSEEN_REGION_Q3_INTERSECTION_CONFIRMATION_PROTOCOL.md`
- Runner: `scripts/v115_unseen_region_q3_intersection_confirmation.py`
- Corrected output: `results/v115-unseen-region-q3-intersection-confirmation-r2/`
- The first run is retained in `...-r1`; r2 corrects only the summary denominator label and preserves the same frozen roster/method.
- EIA file and method hashes are stored in r2 `summary.json`.

# V118 — Broad-Cohort Region-LOO Robustness Diagnostic Results

## Cohort and calibration

The frozen V52 rule selected 43 regions with at least 90% complete-proxy coverage in 2026H1. All 129 Q3 region-month slots were audited. Of these, 124 met the 80% complete-label threshold; 115 also had feasible 2025 lagged capacity constraints. Five slots were coverage-ineligible (BANC September, all three TEC months, and WALC September); nine other eligible slots lacked feasible capacity information. No target outcomes were used to select the region cohort.

For each scored region, the entire target region was excluded from the 25-region source-calibration pool when present, then the frozen V116 all-month maximum-error score and α=10% split-conformal rank were applied. This gives 24 calibration regions/rank 23 for calibration-source regions and 25/rank 24 otherwise. The frozen V115 factor box, physical constraints, and interval intersection were unchanged.

## Coverage and interval widths

| Threshold | Method | Proxy endpoint containment | Region-equal mean width |
|---|---|---:|---:|
| τ=600 | Anchor | 123/124 | 397.6 h |
|  | Monthly-only | 124/124 | 405.0 h |
|  | Capacity, capacity-feasible rows | 115/115 | 260.8 h |
|  | Intersection, capacity-feasible rows | 114/115 | **196.7 h** |
| τ=900 | Anchor | 118/124 | 86.5 h |
|  | Monthly-only | 124/124 | 335.0 h |
|  | Capacity, capacity-feasible rows | 115/115 | 185.0 h |
|  | Intersection, capacity-feasible rows | 109/115 | **44.6 h** |

The intersection cuts width against the physical capacity set by 24.6% at τ=600 and 75.9% at τ=900, but loses proxy endpoint containment in one and six rows, respectively. The narrow intervals therefore do not yet meet the intended “maintain coverage while narrowing” criterion.

## Failure pattern and decisions

At τ=900, only 10 of the 124 eligible region-months had any nonzero high-factor endpoint support; the intersection missed six of those cases. Misses were concentrated in JEA (July and September), LGEE (July), and SC (July–September), where the monthly anchor's upper bound was too small. At τ=600, the single failure was PACE September, with proxy K range [203, 393] hours against an intersection upper bound of 380 hours. In contrast, the capacity-feasible physical set contained every evaluated factor-box endpoint.

The intersection's region-equal 10% tail decision rates were, at τ=600, 7.7% positive / 48.7% negative / 43.6% abstain; at τ=900, 0% positive / 59.8% negative / 40.2% abstain. Some risk decisions improve relative to abstaining anchor intervals, but several negative decisions coexist with uncovered proxy endpoints. These rates cannot be presented as calibrated risk decisions until the anchor failure is fixed and confirmed.

## Conclusion

V118 is a substantive falsification of the current broad claim: the anchor-capacity intersection is efficient on average but undercovers precisely where high-threshold risk is nonzero. Exact capacity-constrained partial identification remains the coverage-safe comparator in this proxy experiment. The next method search should condition calibration on observable coarse risk support (for example, monthly mean-intensity distance to τ and upper-bound support), then evaluate whether it can preserve the capacity set's endpoint coverage while retaining some of the intersection's width gain. The broad Q3 cohort has already been opened, so any such conditional method needs a new region-time confirmation before it can count as a result.

## Reproduction

- Frozen protocol: `V118_REGION_LOO_BROAD_COHORT_PROTOCOL.md`.
- Runner: `../scripts/v118_region_loo_broad_cohort_diagnostic.py`.
- Row-level data, availability audit, per-region frontier, summary and input hashes: `../results/v118-region-loo-broad-cohort-diagnostic-r1/`.

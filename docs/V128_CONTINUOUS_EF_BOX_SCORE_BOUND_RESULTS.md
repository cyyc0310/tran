# V128 — Continuous ±20% EF-Box Score-Bound Results

## Main finding

The conservative score bound accounts for EF vectors inside the box where a source month's mean CIF crosses τ=900, not just the 16 corners. This changes some monthly scores substantially: 14 source-month rows are above their corner-only score, all in SC; SC's maximum score bound rises from 0.523 to 1.000. Yet the leave-one-region-out 90% calibration radius for each of the 37 scored target regions is unchanged from V127 (mean 0.283, range 0.110–0.293). The high SC score is one region-level outlier; the finite-sample 90% rank for other folds is below the maximum.

| τ=900 method | Endpoint coverage | Mean width |
|---|---:|---:|
| Capacity-only | 37/37 | 222.0 h |
| Plain anchor∩capacity | 37/37 | **120.8 h** |
| Frozen q=20% support-gap rule | 37/37 | 144.2 h |

Two fallback rows remain (JEA, PACE). As in V127, both fallback expansions add width without repairing an October endpoint miss. The continuous-box score safeguard therefore does not erase the interval narrowing in this opened cohort, but the observed 37/37 coverage is not an independent coverage guarantee.

## Score-bound logic

For any hourly schedule, each hour's intensity and the monthly weighted mean are nondecreasing in each fossil EF. Let `K_low`, `K_high` be observed event counts at the all-low and all-high EF vectors. If the monthly mean can be below τ, its anchor may be zero and absolute error is bounded by `K_high/T`. If the mean can exceed τ, its anchor may be one and error is bounded by `1−K_low/T`. Taking the maximum over attainable anchor states gives a conservative bound for every independent EF vector inside the box. It can be loose because it does not condition K on the exact EF vector where the monthly mean crosses τ.

The SC result demonstrates that looseness clearly: the bound reaches 1 for months whose mean crosses τ while K spans nearly the whole month across the EF endpoints. The region-level leave-one-out rank absorbs this as a single extreme source region for other target folds; a target fold that excludes SC cannot use SC's own score. The support gate and capacity interval remain essential for this kind of spatial outlier, and more independent regions are needed before interpreting the 90% rank as credible deployment coverage.

## Interpretation

The EF endpoint hull and monthly-mean anchor are monotone, so low/high vectors bound the respective endpoint ranges over the continuous box. The calibration score is different: its interior maximum need not occur at a vertex. V128 uses a conservative upper bound for that score and confirms that the frozen October width frontier does not change after this correction.

This remains a posthoc sensitivity exercise on the October labels already inspected. The method is not yet a validated Chinese predictor. China-facing evidence needs a point-in-time, provenance-audited monthly input panel and independently evaluated hourly labels or a clearly bounded information-identification task. Ember's current data are useful as a secondary data source, but its documentation says monthly provincial fuel splits combine corrected official statistics with other sources and modeled allocation; it is not equivalent to a raw official eight-fuel monthly table.

## Reproduction

- Protocol: [V128_CONTINUOUS_EF_BOX_SCORE_BOUND_PROTOCOL.md](V128_CONTINUOUS_EF_BOX_SCORE_BOUND_PROTOCOL.md).
- Runner: [v128_continuous_ef_box_score_bound.py](../scripts/v128_continuous_ef_box_score_bound.py).
- Row-level source score bounds, target intervals and hashes: `../results/v128-continuous-ef-box-score-bound-r1/`.

# V109 results: independent LP audit of complete K sets

## Outcome

An independent `scipy.optimize.linprog` formulation enumerated every candidate integer `k=0..T` using event-block fuel totals and group-level event/complement capacity inequalities. It was run for all 30 V104 region-months, both thresholds, and low/nominal/high EF vectors: **180 complete-set comparisons**. In every comparison, the LP-derived integer set exactly equals the V92 closed-form enumerator's complete set, not just its endpoints. V104 nominal endpoints match, and low/high sets remain within its reported robust endpoint hull. No label-containment or endpoint discrepancies occurred.

## Strict-threshold boundary audit

The runner records 13,111 cases where the LP objective is within tolerance of `max(0, monthly_excess)`. Inspection shows these are not ambiguous near-events: they all have maximum event excess exactly zero under τ=900 and the low EF vector. The low coal factor is exactly 900 and all other factors are below the threshold, so strict `I_h > τ` events are impossible; the V92 set correspondingly contains no positive K in those cases. The runner's raw “near” counter therefore overcounts numerical ambiguity by treating exact zero as a boundary. The set equality itself is unaffected. Future reports should call these *objective-equality cases* and separately count only positive maxima within tolerance as numerically delicate.

## Interpretation and limitation

This cross-implementation check materially strengthens confidence in the complete-set calculations and strict `>` semantics. It is not a symbolic proof and shares the same underlying model assumptions; a publication still needs the algebraic theorem, a transparent derivation, and preferably a second solver or exact rational optimization audit. It validates only the EIA fixed-factor proxy identified sets, not real CIF or forecast calibration.

## Reproducibility

- Frozen protocol: `docs/V109_INDEPENDENT_LP_SHARP_SET_AUDIT_PROTOCOL.md`
- Runner: `scripts/v109_independent_lp_sharp_set_audit.py`
- Outputs: `results/v109-independent-lp-sharp-set-audit-r1/`
- Hashes and source audits are in `summary.json`.

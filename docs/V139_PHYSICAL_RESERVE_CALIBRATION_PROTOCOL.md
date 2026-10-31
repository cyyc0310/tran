# V139 — Physical-Reserve Calibration under Complete Monthly Generation Margins

## Question and input contract

The user confirms that monthly generation by all categories is available. Treat full monthly fuel margins as the primary information budget. Three-day summaries remain an optional-information experiment. This development round asks whether source-region calibration of the remaining physical tail-risk space improves on a uniform anchor radius and the existing support-gap fallback.

Reuse V120 r2's 589 region-months from 25 previously examined EIA source regions, two thresholds (600/900), and its ±10% fuel-factor box. This is a different EF contract from V129–V138; do not compare their absolute widths as paired effects. Existing source labels are robust K endpoint pairs. Capacity is a U.S. proxy and is not yet a confirmed Chinese input.

Pre-scoring amendment: the first preparation audit stopped because 44 archived physical intervals exclude their corresponding robust K endpoints. A join to V100's pre-outcome monthly-energy availability table identifies 76/589 capacity-infeasible months, including all 44 conflicts. The remaining 513 months' 1,026 threshold endpoint pairs exactly match V103's audited output. Do not exclude cases by label agreement. Retain all 589 months, use [0,T] when monthly energy violates the capacity contract, and calibrate all methods on this same corrected information budget. Save the archived conflict audit. Fix the solver's missing global feasibility check, which previously allowed endpoint shortcuts to return (0,) or (T,) for infeasible monthly inputs. Record minimal boundary examples in the run output. The archived V120 fallback is invalid as an overall performance comparator until its input availability is corrected.

## Frozen comparison

For every outer held-out region, remove that region entirely from calibration. Base anchor endpoints are the uncalibrated monthly-constant factor-box endpoints in V111. Let [L,U] be the exact physical K endpoint hull, and [a,b] the raw anchor in hours. Project [a,b] into [L,U] to obtain [a*,b*].

Compare:

1. Physical hull [L,U].
2. Raw anchor absolute error in monthly fractions, group-calibrated then intersected with the physical hull; empty intersections fall back to [L,U].
3. Projected-anchor absolute error in monthly fractions, with the same calibration rule and clipping.
4. Physical-width scaled error: max(a*−K_low, K_high−b*, 0)/(U−L), with zero score for singleton physical hulls.
5. Directional-reserve score: max((a*−K_low)_+/(a*−L), (K_high−b*)_+/(U−b*)), with 0/0=0. Invert it as [a*−q(a*−L), b*+q(U−b*)].

For every score, first take the maximum over months within each calibration region. Use finite-sample rank ceil((n_regions+1)(1−alpha)) with alpha in {0.05,0.10,0.20}; if rank exceeds available regions, return the physical hull. Report the entire frontier and do not choose a winner using the target outcomes. For symmetric width scaling, expand both sides by q(U−L), then clip to the physical hull. Outward integer endpoint conversion preserves every attainable integer in the continuous interval. Calibrate every method on the same full cohort with availability fallback. Report capacity-feasible and capacity-unavailable subsets separately. The archived V120 table is retained only as an explicitly invalidated historical reference.

## Audits and endpoints

Save calibration membership/rank/radius, fold rows and hashes. Assert no target region in its calibration pool, complete denominator preservation, physical containment of every robust label, nonempty intervals, interval subset of the physical hull, score bounded by one for the physical normalized methods, and expansion as alpha decreases for projected-anchor methods. Report any raw-anchor nesting failure separately: its empty-intersection fallback may switch from a full physical hull to a partial intersection as the radius grows. Report region-equal monthly coverage, all-month region coverage, width, conditional coverage for possible positive tails (K_high>0) and strictly positive robust tails (K_low>0), and positive/negative/abstain/error at ≥1/5/10% frequencies. Compare width-scaled and directional methods for exact output equivalence rather than counting redundant methods as separate innovations.

## Scientific boundary

Normalized conformal scores and inference for interval-valued outcomes already exist. This experiment can establish an efficiency mechanism in this CIF information contract; it does not by itself establish methodological novelty. Region exchangeability is an assumption, not validated by leave-one-region-out empirical coverage. These are retrospective target-month statistics and proxy labels, not prospective Chinese hourly forecasts or blind confirmation.

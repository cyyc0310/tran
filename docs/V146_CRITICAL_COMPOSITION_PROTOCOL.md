# V146 — Joint Risk-Curve Calibration in Equivalent Composition Coordinates

## Mechanism hypothesis

V145 shows that ordinary recomposition plus residual CQR can repair LGEE/600, but residual calibration remains fragile at 900. Test a specific alternative to adding vertical K-error bands: invert the source-profile risk-response curve and calibrate an equivalent horizontal coal-share displacement, jointly over thresholds 600/900 and both factor endpoints.

This displacement is a model-mismatch coordinate. The observed monthly margins remain exact; it is not measurement error, a causal coal intervention or an identified latent physical parameter. Generic conformal inversion is prior art. This is a mechanism probe, not an established novelty claim.

## Frozen setting

Reuse V144's canonical factors/profiles, V145's PACE/LGEE/SC target months, 30-fit/12-calibration roles, two seeds and alpha=0.10/0.20. No new region/time is opened. No target factor-floor or target-hour feature is allowed.

For every query retain the V145 conditional donor identities and weights at its observed composition. Hold these weights fixed during the response sweep; otherwise kernel reweighting could destroy monotonicity and confound the mechanism. Keep the query's noncoal fuel ratios fixed and vary virtual coal share p in [0,1), with p_max=nextafter(1,0). If the query were pure coal, use a declared physical-hull fallback; the frozen cohort has no such query.

For a donor hour, excess intensity has sign p A+(1−p)B, where A=(e_coal−tau)q_coal≥0 and B is the query-noncoal-ratio-weighted donor excess. Its critical coal share is −B/(A−B) when B<0 and A>0. B>0 yields an always-event hour on [0,1); B=0,A>0 gives a strict crossing at p=0; A=0,B≤0 yields no event. Thus donor tail fractions and their fixed-weight quantiles are nondecreasing in p for the declared thresholds/factors.

Use delta grid {0,1/256,...,1}. Lower response bound is the weighted 0.05 low-factor quantile at max(0,p_observed−delta); upper bound is the weighted 0.95 high-factor quantile at min(p_max,p_observed+delta). Intersect with the actual observed-margin physical hull when producing output intervals. At delta=0 this must reproduce the V145 conditional initial band exactly.

## Scores and ablation

For each calibration month, choose the smallest grid delta covering its saved K_low/K_high at both thresholds. Take maximum over months within each calibration region, then the usual finite-sample region rank. A score is infinity if no grid value covers it; an infinite selected radius returns the actual physical hull, and the fallback is counted. Do not hide support failures by changing the response endpoint convention.

Compare two joint-calibration methods:

1. Horizontal critical-composition envelope as above.
2. Vertical-only control: use the same delta grid but linearly interpolate the delta=0 initial bounds toward the actual physical hull at delta=1. Calibrate the same joint endpoint/month/region score. This controls for replacing absolute errors with an arbitrary normalized expansion family.

Retain V145's ordinary conditional recomposition, common/response-informed CQR, residual CQR and anchors as archived same-budget references. Their original per-threshold calibration differs from the new joint score and must be labelled. Report joint two-threshold coverage as well as individual coverage, width and interval score.

## Audits and interpretation

Build critical curves from saved source profiles only. Resolve near-zero noncoal excess with rational arithmetic. At the observed query composition, match every selected donor's low/high count to the exact V144 response cache; correct floating critical-share ties by exact sign. Save the number of corrections. Assert delta nesting, fixed weights, no calibration/target donor, complete denominator, and physical subset. Save curves, inverse scores, selected radii, support fallbacks, roles and hashes.

Region-level conformal validity requires exchangeability of whole region objects under a fixed proper-source predictor; it is not an arbitrary cross-country guarantee. This probe calibrates a nested response family, not true coal shares or hidden dispatch parameters. Current search found generic inverse conformal work including [LatentCP, 2026](https://arxiv.org/abs/2608.03607), and multi-parameter calibration including [MCP, revised October 2026](https://arxiv.org/abs/2605.12341); only their primary abstracts were checked here, so no theorem-level novelty comparison is complete. A positive result must beat the vertical control and strong V145 methods, then undergo a full literature/proof review before any freeze for unopened confirmation.

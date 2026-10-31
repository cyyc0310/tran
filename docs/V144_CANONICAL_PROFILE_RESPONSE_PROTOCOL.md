# V144 — Canonical Factors and Source-Profile Risk Responses

## Input preparation, not a new method claim

Reuse the 936 region-months / 43 already examined identities from V140 and eligible V141 additions. Do not add a region or time period. Build an immutable cache of their valid 2024–25 hourly fuel generation, exact monthly fuel margins, offsets and monthly-only features. Reproduce all archived hours, monthly shares and legacy EF-box endpoint counts before changing a contract.

Canonical factors use integer literals: nominal (1000,400,800,0,0,0,0,400), low (900,360,720,0,0,0,0,360), high (1100,440,880,0,0,0,0,440), in the existing eight-fuel order. These are the intended exact 9/10 and 11/10 factors, not estimates of Chinese uncertainty. Recompute both thresholds 600/900 with exact strict comparisons, save every difference from the archived floating-factor contract, and preserve old outputs. No improvement may be credited to this numerical contract change.

## Risk-response operator

For donor month d and query monthly fuel margins M, define normalized donor shapes q_dfh=T_d x_dfh/M_df when M_df>0 and q_dfh=1 otherwise. Under uniform sampling of every valid donor hour, the mean of each q is one. The hypothetical intensity is sum_f e_f M_f q_dfh / sum_f M_f q_dfh. Assign zero-output counterfactual hours zero intensity. Its strict-threshold count fraction is the source-profile risk response R_d(M,e,tau).

This is the existing fuel-recomposition idea evaluated as a monthly tail response. It assumes a donor's normalized profile can be reused under a new composition; it is not a causal intervention label, a sharp physical identified set or a new algorithm. Uniform complete-donor-month normalization conserves fuel margins in the time-normalized donor law; it does not certify a realized T_query-hour sequence when donor/query valid-hour counts differ. Missing-fuel flat profiles are assumptions and their query mass must be reported.

Cache R for every query monthly-margin row and every donor profile at the low/high canonical factors and both thresholds. Query rows may expose only their monthly margins; their own hourly values are used only in the explicitly separated identity audit. Later fitting must subset donor identities before extracting features, excluding calibration and target identities. Preserve exact donor/query indexing and hash the response cache.

## Numerical and information audits

Use a floating dot product with an explicit roundoff bound to detect potential zero-boundary cases. Resolve nonstructural near-boundary signs with exact rational arithmetic using the saved raw generator values, exact monthly margins and literal factors. Structural zero excess is non-event. Compare each donor's own exact composition response to its canonical count, verify all-low/all-high response ordering and count ranges, and independently check a deterministic grid of cross-region donor/query cases with the full rational implementation. Save the checks, near-boundary correction counts, raw hashes, scripts and protocol hash. Fail on mismatch.

The source response is a predictor ingredient. Oracle target factor-floor values from V143 must never be passed as query features or used for model selection. This phase evaluates no unseen-region claim and does not satisfy the original hourly forecasting gate.

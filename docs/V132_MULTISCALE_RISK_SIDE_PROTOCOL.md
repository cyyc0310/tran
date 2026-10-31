# V132 — Multiscale Risk-Side Information Budget

## Question

What is the coarsest local reporting interval that retains most of the tail-count identification gain from daily threshold-side summaries?

## Frozen design

- Reuse the V129 frozen 358 region-month EIA cohort, proxy factors and factor-support intervals, τ=600/900, and ≥1/5/10% count decisions. No target outcomes are used to select a bin size.
- Compare reporting blocks of 1, 3, 7, and 14 UTC calendar days within each target month. The month boundary is never crossed. Use identical valid hourly observations across all block sizes.
- Every block reports three aggregates: total thermal generation, generation on the threshold high side, and total clean generation. No capacity is used. The monthly risk-side contract is the coarse baseline; block detailed fuels are an oracle upper-information comparator.
- Compute robust risk-side and detailed-fuel intervals on each block, then sum integer lower/upper endpoints. This is conservative under a common unknown factor across blocks.
- Use a closed-form no-cap risk-side feasibility rule (support-total checks for K=0/K=T and a positive high-side plus low-side remainder for interior K) to avoid enumerating every within-block count. Cross-check the closed form against the general V129 risk-side solver on a fixed synthetic grid before scoring. The detailed-fuel no-cap oracle uses V90's fixed-factor set at the low and high EF endpoints, matching V129's robust endpoint construction.
- Preserve the fixed monthly denominator and report valid block counts, scalar counts, width, selective decisions, oracle gap, and nesting failures.
- Assert each binned interval contains the summed nominal fixed-factor K and the detailed-fuel robust envelope for the same bins. No row is silently dropped.

## Interpretation boundary

The experiment measures information value under retrospective oracle coarsening of U.S. EIA hourly generation. It does not validate Chinese reporting feasibility, release timing, reporting errors, or actual hourly CIF prediction. The selected frequency, if any, is a candidate data-collection contract, not a deployment recommendation.

# V125 — October-Matched Calibration Diagnostic

## Question

How sensitive is the V122 October MTD result to using a full-year source score maximum instead of an October-matched source score?

## Diagnostic procedure

- Reuse V122's already opened target rows and exact factor-box endpoint labels. This is explicitly post-hoc, not a new confirmation.
- For each target region and threshold, take 2024/2025 October source scores only, exclude the target region if it is in the source calibration pool, aggregate each remaining region's monthly worst-case error by maximum, and use split-conformal α=0.10 with rank `ceil((n+1)*0.90)`.
- Reconstruct the target's binary monthly-mean anchor base endpoints from the V122 interval and radius, then recompute anchor and anchor∩capacity endpoints with the October-matched radius. Keep capacities, support-gap features/boundaries, and eligibility unchanged.
- Apply the same frozen policy: τ=600 capacity-only; τ=900 q=20% support-gap fallback to capacity, otherwise anchor∩capacity.
- Compare radius, coverage, widths, fallback rate, and decision frontier to V122's full-year calibration.

## Interpretation boundary

The V122 target outcomes have already been observed; this tests calibrator sensitivity only. October calibration uses two full historical Octobers while the target is a partial October, so the horizon match remains imperfect. A new untouched time block is required before adopting this calibration variant.

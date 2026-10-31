# V119 — Source-Support Gap Fallback Diagnostic Results

## Frozen rule result

Using 2024–2025 source data, exact ±10% EF-box capacity-constrained K upper endpoints, and leave-one-region-out anchor calibration, the source-only maximum normalized disagreement was 0.4798 at τ=600 and 0.9785 at τ=900. Applying the frozen rule “fallback to the physical capacity interval only when target disagreement exceeds this source maximum” flagged no V118 target rows. Width and endpoint coverage therefore remained unchanged: intersection coverage was 114/115 and 109/115 at τ=600/900, respectively. The maximum-support alarm did not solve the failures.

## Post-hoc threshold sensitivity

As a diagnostic only, we also varied the source-region quantile used as the support boundary, with each target region excluded from its own boundary calculation. At the 80th percentile boundary (rank 20 for n=24, rank 21 for n=25), the fallback rule restored all factor-box endpoint containment on the opened V118 cohort. Region-equal mean widths were 199.5 h at τ=600 and 84.1 h at τ=900, compared with capacity-only widths of 260.8/185.0 h. At the 90th percentile boundary, coverage remained 114/115 and 109/115. The 80th percentile was identified by a post-hoc sensitivity sweep on the same opened target cohort; it is not confirmatory evidence and must not be reported as an independently selected method.

## Interpretation

The feature itself is promising: disagreement between monthly anchor upper support and physical capacity upper support points toward the observed high-risk failures. But the source maximum is too permissive, and selecting the 80th percentile after seeing target failures risks tuning to this quarter. The next legitimate step is source-only region cross-validation over candidate support quantiles, then freeze the quantile and evaluate it on an untouched region-time cohort. Until then, the safe proxy comparator remains the capacity-constrained sharp set.

## Reproduction

- Frozen protocol: `V119_SUPPORT_GAP_FALLBACK_PROTOCOL.md`.
- Runner and outputs: `../scripts/v119_support_gap_fallback_diagnostic.py`, `../results/v119-support-gap-fallback-diagnostic-r1/`.

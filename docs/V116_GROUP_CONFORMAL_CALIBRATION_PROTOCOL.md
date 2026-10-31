# V116 — Region-Grouped Split-Conformal Calibration Diagnostic

## Question

Can a finite-sample, region-level split-conformal radius replace the empirical worst-source-region monthly-q90 radius used by V112/V115, while preserving the V115 factor-box proxy containment and narrowing the anchor-capacity intersection?

## Frozen procedure

- Calibration source: the unchanged V111 `source_calibration_rows.csv`, comprising 25 calibration regions and monthly scores for 2024–2025.
- Score for each calibration region and threshold: the maximum `worst_case_absolute_error` over all available calibration months in that region. This makes the score region-level and protects an entire calibration block rather than treating correlated months as independent.
- For each threshold separately, use split-conformal miscoverage `alpha=0.10`; with `n=25`, use the one-indexed order statistic `ceil((n+1)*(1-alpha)) = 24`.
- Apply that radius symmetrically to the V115 anchor's factor-box base fraction, clip to `[0,1]`, convert to integer K endpoints, and intersect with the same V115 sharp capacity set. Do not change the target roster, capacity constraints, factor box, eligibility rule, or decision thresholds.
- Compare to V112's frozen radius, the monthly-only factor-box set, the capacity set, and their V115 intersection.

## Interpretation boundary

This is an exploratory recalibration diagnostic on V115 labels already opened while designing the intersection. It is not a blind confirmation. The split-conformal statement is marginal over exchangeable regions under the usual split-conformal assumptions; it does not guarantee conditional coverage for a particular region, nor validate the EIA proxy against observed Chinese hourly CIF. Any apparent gain must be frozen and confirmed on a new region cohort before being presented as held-out evidence.

## Frozen inputs

- V111 calibration rows: `results/v111-factor-robust-monthly-anchor-r1/source_calibration_rows.csv`.
- V115 target roster and source inputs: frozen in `docs/V115_UNSEEN_REGION_Q3_INTERSECTION_CONFIRMATION_PROTOCOL.md` and recorded by V115 output hashes.
- V115 target labels are already observed; this protocol does not claim test-set blindness.

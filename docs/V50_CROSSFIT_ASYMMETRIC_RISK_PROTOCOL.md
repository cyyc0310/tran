# Cross-fitted asymmetric tail calibration after monthly location correction

Registered 2026-10-27 before computation. The preceding cross-fitted risk comparison found an interval-score/width gain from EF-only monthly level correction, but WACM remained undercovered almost entirely in the upper tail. This tests whether separately calibrated lower/upper expansions reallocate unused tail width.

## Fixed design

- Reuse the same V50 frozen seals, 70 external prediction packages, ten external labels, date-local three-month monthly mix, 90-day support, and three frozen point arms (Scale-anchor, B0 AlignedKeep, B1 AlignedKeep) with MCStd, HETScale, and CQRplus. Verify all hashes; no target/unseen artifacts and no model-head refits.
- Use two fixed center procedures: unshifted, and EF-only daily location correction. The EF-only correction is cross-fitted: for each calibration source, train on the other eight sources; for the held evaluation source, train on the other nine. Exclude fossil-increment term to isolate location/tail asymmetry.
- For each held source, compute calibration scores from the nine cross-fitted calibration sources. Compare (a) existing equal-region 90% symmetric conformal score `max(lower_excess, upper_excess)` and (b) separate equal-region 95th percentiles of nonnegative lower and upper excess scores. Apply the same held-source correction before expansion, multiply q by the frozen method scale, then clip bounds to [0,1000].
- Report source-level and macro coverage, width, interval score, lower/upper misses, and q values. Unit is source; days/outer functions are repeated measures.

## Interpretation boundary

Asymmetric tail calibration is a standard risk baseline, not a new conformal method. This is a descriptive source-held-out diagnostic on ten reused development sources; it cannot guarantee per-source coverage or establish China transfer. No unseen source is opened.

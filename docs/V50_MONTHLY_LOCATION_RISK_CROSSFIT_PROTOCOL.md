# Cross-fitted monthly location correction with sealed risk baselines

Registered 2026-10-27 before computation. The prior experiments improved level CRPS but used a fixed factor-2 interval. This tests whether legal monthly location correction can improve intervals built by frozen MCStd, HETScale, and CQRplus risk arms when calibration scores are themselves cross-fitted through the correction.

## Fixed scope

- Reuse the V50 freeze and prediction/calibration seals; 70 external prediction packages; ten external labels; 90-day eligibility; and strict date-local three-month eight-fuel shares. Use only the frozen point arms Scale-anchor, B0 SizeET AlignedKeep, and B1 SizeCapRidge AlignedKeep. Risk methods are MCStd, HETScale, and CQRplus. Recheck package/input hashes. Never open target-role predictions/scores, PACE/LGEE, or unseen sources. Do not refit point or risk heads.
- Compare three location procedures: no shift; EF-only correction; EF plus a fixed λ=.25 orthogonal fossil-share increment. λ=.25 is carried forward from prior development rounds and is **not tuned in this experiment**. Fit source-equal weighted daily residual regressions on source sets described below.
- For each outer point function × point arm × held external source, use the other nine sources as calibration/training pool. For every calibration source in that pool, fit each correction on the other eight sources, then apply it to the calibration source's sealed risk bounds. Compute hourly nonconformity scores from those shifted bounds and the stored scales. This cross-fitting prevents a source's own hourly labels from fitting its location correction and then calibrating that correction's score.
- For each of the three location procedures, calibrate an equal-source/hour 90% conformal q from the nine cross-fitted calibration sources using `risk_baselines_v50.calibrate(..., rule='equal_region')`. Fit the held-source correction on all nine training sources, shift its stored initial bounds, expand by the corresponding q×scale, clip to [0,1000], and evaluate hourly coverage, width, interval score, and tail misses.
- Report per-source and macro results for all 3×3×7×10 outer configurations. Source is the independent unit; outer functions and hours are repeated views. No significance claims.

## Interpretation boundary

This is development source-held-out evidence on ten previously used regions. It tests location correction with conformalized frozen risk heads but does not guarantee conditional coverage, prove China transfer, or establish method novelty. A favorable result would still require a frozen method and genuinely unseen-source validation.

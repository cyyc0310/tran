# V112 protocol: worst-source-region calibrated robust anchor

Frozen 2026-10-28. Reuse V111's factor-robust anchor, source calibration residuals, and exact V104 target cohort. Replace only the region-balanced pooled q90 radius with the maximum across source regions of each region's empirical 90th percentile of worst-case factor-box residual. Use the same [0,1] clipping, integer endpoint conversion, tail thresholds, and V105 partial-set comparators.

This deliberately conservative transfer baseline tests whether V111's two FPC misses were caused by pooled calibration. It must report coverage, width, and selective decisions without retuning or excluding regions. The max-region quantile is empirical and is not a formal guarantee for a new region. Same EIA fixed-factor proxy, factor stress box, retrospective window, and regional dependence limitations apply.

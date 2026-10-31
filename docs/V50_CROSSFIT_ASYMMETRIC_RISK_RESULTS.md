# V50 cross-fitted asymmetric risk: development result

Run date: 2026-10-27. Protocol: [V50_CROSSFIT_ASYMMETRIC_RISK_PROTOCOL.md](V50_CROSSFIT_ASYMMETRIC_RISK_PROTOCOL.md). Output: `results/disjoint-calibration-v50-crossfit-asymmetric-risk/`.

## Design and scope

The registered comparison reused the V50 frozen inputs: 70 outer-by-source prediction packages, ten external label packages, seven outer functions, and ten regions. For each held region, the monthly EF-only level correction was fit on the other nine regions. Calibration-region corrections were themselves cross-fitted on the other eight regions before their hourly scores entered calibration. Point and risk heads were not refit; no target-role package, PACE/LGEE package, or unseen-region score was opened. The run reports 2,520 held-source rows and 2,100 location fits. The summary records the freeze, prediction, calibration, and monthly-input hashes.

This is development evidence on ten previously used regions. It is not an independent regional confirmation, and the seven outer functions and hourly observations do not increase the independent region count.

## Main comparison

Selected development arm: `B0-SizeET-AlignedKeep + MCStd`. Metrics below are equal-region means; each region first aggregates its seven outer-function views.

| Location and tail rule | Coverage | Mean width | Interval score | Worst region coverage |
|---|---:|---:|---:|---:|
| No shift, symmetric 90% | 86.97% | 637.74 | 843.67 | 51.12% |
| EF-only shift, symmetric 90% | 95.47% | 475.15 | 512.83 | 83.08% |
| EF-only shift, separate 95th-percentile tails | 93.82% | 415.57 | 447.47 | 86.52% |

Separate-tail calibration improves over EF-only symmetric calibration by 59.58 width units (12.5%) and 65.36 interval-score units (12.7%). Macro coverage falls 1.66 percentage points, while the worst-source coverage rises from 83.08% to 86.52%. Against the unshifted symmetric MCStd comparator, the selected row is 34.8% narrower and has a 47.0% lower interval score, with higher macro coverage. These are observed development-fold contrasts, not inferential estimates.

Source-level coverage for EF-only symmetric → EF-only separate-tail is:

| Region | Symmetric | Separate tails | Width (symmetric → separate) |
|---|---:|---:|---:|
| SWPP | 94.11% | 86.52% | 493.37 → 500.66 |
| AVA | 98.75% | 96.35% | 371.78 → 359.10 |
| BANC | 99.90% | 97.20% | 370.89 → 356.79 |
| JEA | 99.88% | 99.19% | 677.45 → 495.96 |
| SC | 96.36% | 98.08% | 519.25 → 477.54 |
| NWMT | 93.10% | 95.86% | 880.75 → 681.95 |
| SCL | 100.00% | 88.75% | 9.00 → 11.69 |
| WACM | 83.08% | 93.63% | 499.50 → 433.24 |
| SEC | 94.22% | 88.11% | 588.09 → 532.72 |
| SCEG | 95.35% | 94.49% | 341.42 → 306.02 |

The change is a tail reallocation, not a uniform widening: WACM's upper-tail miss rate falls from 16.88% to 6.32%, while lower-tail misses rise for several regions. It is not uniformly better: SWPP and SEC lose more than five coverage points, and their separate-tail coverage is 86.52% and 88.11%. Thus the minimum-source improvement does not mean source-conditional reliability has been solved.

## Robustness and interpretation

SCL's very small intervals materially affect the equal-region macro width and score. Excluding SCL as a **sensitivity diagnostic only** (not as a post-hoc source exclusion), EF-only symmetric calibration gives coverage 94.97%, width 526.94, score 568.81; separate-tail calibration gives 94.38%, width 460.44, score 495.67, and minimum-source coverage 86.52%. The efficiency gain persists without SCL (width −12.6%, interval score −12.9%), but this remains a ten-source development comparison. SCL is retained in all primary figures.

The selected configuration was identified after inspection of a 3 point-arm × 3 risk-method development grid. Accordingly, it is a candidate configuration, not a confirmatory estimate. The 95th percentile in each tail is a standard asymmetric conformal construction; quantile-based adaptive intervals are established baselines, and recent work directly studies skew-adaptive conformal intervals. This experiment supplies no algorithmic novelty claim. See [Conformalized Quantile Regression](https://doi.org/10.48550/arXiv.1905.03222) and [Skew-adaptive conformal prediction](https://doi.org/10.48550/arXiv.2605.16145).

## Decision

The combined EF-only center correction and separate-tail calibration was frozen for one unseen-region evaluation because the development folds showed a clear aggregate efficiency gain and the worst-source gate passed. The preselected TVA 2026H1 confirmation then failed: coverage was 77.67%, all misses were lower-tail, and interval score was worse than both the matched symmetric EF-only MCStd and the simple factor-2 anchor. See [V50_TVA_ASYMMETRIC_CONFIRMATION_RESULTS.md](V50_TVA_ASYMMETRIC_CONFIRMATION_RESULTS.md). The method is rejected for transfer; its development gain does not support an algorithm novelty claim or China transfer.

The repository's prepared external-v6 files contain the six previously studied targets AZPS, DUK, LGEE, PACE, PSCO, and SRP. The separate V6 metadata audit names TVA as the seventh qualifying candidate, with 181 complete 2026H1 days; a prior inventory records no TVA score/prediction artifact and says only quality metadata had been inspected. Based on the explicit prior instruction to freeze after a clear development increment, the fixed TVA 2026H1 confirmation is now registered in [V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md](V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md). The target scores remain unopened; confirmation is pending a sealed seven-outer × 181-day prediction run. A passing TVA result would be one new-region success, not proof of general or China transfer.

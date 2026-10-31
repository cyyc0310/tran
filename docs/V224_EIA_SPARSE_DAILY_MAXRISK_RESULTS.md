# V224 Sparse local day profiles improve hourly risk intervals on EIA proxy data

Date: 2026-10-30. Valid run: r3. r2 was retired after a timestamp-resolution bug affected the daily grouping and simultaneous-day coverage; do not cite that folder.

## Results

There are 24 target balance areas and 135 eligible region-months in 2026H1. The source-only, leave-target-region-out interval reaches 95.64% hourly coverage, width 267.75 gCO2/kWh, and Winkler-90 295.89. Its simultaneous coverage of all hours in a random unobserved day is only 83.58%, with the worst region at 22.9% (NWMT).

| Random local full-day profiles | Hour coverage | Simultaneous day coverage | Width | Winkler-90 |
|---:|---:|---:|---:|---:|
| 3, 5, 7 | 100%, unbounded radius | 100%, unbounded radius | ∞ | ∞ |
| 9 | 98.64% | 92.58% | 213.18 | 218.47 |
| 12 | 99.05% | 94.32% | 222.37 | 225.82 |
| 15 | 99.28% | 95.38% | 228.42 | 230.84 |

For 9 days, source-relative hourly coverage rises by 3.01 percentage points [region bootstrap 95% CI 0.63, 6.02]; simultaneous day coverage rises by 9.01 points [1.41, 17.99]. Winkler-90 drops 77.42 [24.32, 133.32]. The interval width falls 54.58, but its confidence interval [−119.83, +9.97] crosses zero. Nine areas still have average simultaneous day coverage below 90%; the worst is AVA at 88.93%.

At 12 days, all 24 regions' average simultaneous day coverage exceeds 90% (worst TEC at 91.78%). Macro coverage is 94.32%; Winkler improvement vs source-only is 69.86 [region-cluster 95% CI 14.49, 126.48]. Width improvement is 45.39, but the CI crosses zero [−114.00, +24.52]. This makes 12 days a more stable candidate in this tested grid, while 9 remains the first finite conformal-rank budget and the narrowest/Winkler-best among finite radii.

## Combined reading with V222–V223

V222 found no daily-shape transfer gain from monthly fuel margins on Carbon Monitor. V223 found that random daily fuel snapshots can calibrate daily CIF-proxy risk intervals, with 12 points a promising tested budget on the same secondary dataset. V224 now shows a corresponding risk-only value on EIA's hourly fuel-generation proxy: a random local day must contribute its full hourly fuel curve; nine day-level maximum scores are the finite-rank threshold, and twelve gave stronger region-average simultaneous-day coverage in this run.

The operational information requirement is substantial: 12 complete days × 24 hours × 8 fuel categories per target region-month, available only after observing the target month. This information is absent from the user's described monthly tables. Thus the result motivates an information-collection study but does not provide an immediately deployable China method.

## Novelty and limits

The radius is split conformal, and whole-day max scoring is a standard cluster-level calibration idea. Design-based conformal already treats finite-population sampling; a 2026 preprint treats conformal regions for continuous trajectories observed at sparse random times. [Design-based conformal prediction](https://www150.statcan.gc.ca/n1/pub/12-001-x/2023002/article/00007-eng.htm) and [randomly sampled trajectory conformal regions](https://arxiv.org/abs/2608.29559) make a generic conformal novelty claim untenable.

The remaining possible contribution is a CIF-specific information budget benchmark that separates monthly level identification from local temporal risk calibration, and establishes which legally obtainable target-side measurements reduce interval risk. A paper claim still needs an independent operational data family and a public China information route. EIA proxy results are not independent of prior V215–V218 work.

## Files

- [Protocol](V224_EIA_SPARSE_DAILY_MAXRISK_PROTOCOL.md)
- [Runner](../scripts/v224_eia_sparse_daily_maxrisk.py)
- Valid outputs: `results/v224-eia-sparse-daily-maxrisk-r3/`

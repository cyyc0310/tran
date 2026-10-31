# V50 monthly mix volatility does not add a reliable source-risk signal

Sol, 2026-10-27. This source-held-out diagnostic follows the prior positive but small-sample full-mix level association. It tests one fixed hypothesis: whether recent legal month-to-month mix movement carries additional information about standardized hourly risk beyond the three-month mix level.

## Protocol and data

The registered [protocol](V50_MONTHLY_MIX_VOLATILITY_PROTOCOL.md) reused the ten sealed V50 source-level p90/p99 risk outcomes and the two original hash-bound monthly CSVs. For each July–December 2024 query date, it selected the three most recent complete months whose period and release were legal at that date. The level is the fixed-EF-weighted monthly share proxy averaged across those contexts; volatility is the mean adjacent absolute change within each three-month window, averaged across query dates. It used four fixed LOO models (null, level, volatility, level+volatility), with ten sources as the independent units. No production prediction or risk model was refit.

The complete 10-source feature and prediction tables and hashes are in `results/disjoint-calibration-v50-mix-volatility-probe/`. The level feature was checked against the already sealed V50 full-mix probe and matched within `1e-9`. This remains a project-EF proxy with simulated release times, not a Chinese accounting factor or verified Chinese release vintage.

## Results

| Outcome | Model | LOO MAE | LOO RMSE | Sign accuracy for source deviation |
|---|---|---:|---:|---:|
| p90 | Null mean | 2.009 | 2.154 | — |
| p90 | Level only | **1.498** | **1.767** | 70% |
| p90 | Volatility only | 1.861 | 2.315 | 80% |
| p90 | Level + volatility | 1.601 | 1.754 | 70% |
| p99 | Null mean | 2.682 | 2.831 | — |
| p99 | Level only | **2.064** | **2.414** | 70% |
| p99 | Volatility only | 2.726 | 3.377 | 80% |
| p99 | Level + volatility | 2.355 | 2.727 | 70% |

Volatility alone is worse than the null for both p90 and p99 MAE/RMSE. Adding it to level worsens MAE; p90 RMSE improves by only 0.0128 while p99 RMSE worsens. That isolated p90 RMSE change does not establish incremental signal across only ten previously used domains, particularly since the p90 MAE and the full p99 comparison move in the wrong direction.

## Interpretation

The current monthly full-mix **level** proxy remains the only tested coarse feature with a clear source-held-out association to these risk summaries. A single volatility feature does not improve the prediction robustly, and no interval calibration or coverage experiment follows from this result. Do not add this feature to the candidate risk model or claim an algorithmic contribution. The result does not prove that all temporal features are useless; it rejects this fixed mean-absolute-change statistic under this data/proxy/ten-source setup.

No target-role package, PACE/LGEE, 2025 response or unseen source was opened. No method was frozen. The next value lies in deciding whether the level signal can be made useful without violating coverage, not in expanding the same ten-source feature set.

# V393: release-lag composition-state identifiability

## Question

Can public monthly generation composition support a credible *ex-ante* local threshold classification when the target month has not started? This is a data-timing and mean-support identification check. It is not an hourly CIF prediction experiment and it does not measure predictive interval coverage or sharpness.

## Information timing

The NBS energy releases show why same-month or immediately preceding-month composition cannot be treated as known at month start. NBS published the combined January–February 2026 report on March 16, the March report on April 16, and the April report on May 18. Thus, the prior month's report arrives during the target month. For a forecast issued on day 1, this audit uses calendar month `t-2` as the latest monthly observation available. A `t-1` lag is included only as a release-after-start sensitivity, never as a month-start feature.

Sources: [NBS January–February 2026 release](https://www.stats.gov.cn/sj/zxfbhjd/202603/t20260316_1962787.html), [NBS March 2026 release](https://www.stats.gov.cn/sj/zxfb/202604/t20260416_1963324.html), [NBS April 2026 release](https://www.stats.gov.cn/sj/zxfbhjd/202605/t20260518_1963725.html).

The regional monthly API snapshot is a current historical extract, not an as-published vintage archive. The release dates establish the timing convention, but this does not prove that each historical value was identical to the value known at its original release date.

## Data and method

- Input: five deliberately selected provinces (Shanghai, Sichuan, Jilin, Fujian, Qinghai), official monthly total generation and thermal generation, 2018-01 through 2026-09.
- The target state exactly matches the V392 binary cross-threshold certificate: `400*s <= 300 < 1000*s`, where `s` is thermal share. This is true for `0.30 < s <= 0.75`; outside that band, the assumed mean-support interval lies wholly to one side of 300 g/kWh.
- Missing monthly values remain missing. January–February aggregates are not split into monthly values. The `t-2` test has 345 province-month cases; the `t-1` sensitivity has 390.
- The national comparator pools only these five selected provinces. It is a diagnostic pooled anchor, not a representative China-wide national estimate or a strong forecasting baseline.

Reproduce with:

```bash
python scripts/analyze_v393_release_lag_identifiability.py
```

## Results

| Information available | State agreement | Sensitivity | Specificity | N | State changes |
|---|---:|---:|---:|---:|---:|
| Local composition, t-1 (mid-target-month sensitivity) | 96.2% | 94.7% | 96.9% | 390 | 15 |
| Five-province pooled composition, t-1 | 34.1% | 98.5% | 1.2% | 390 | 257 |
| Local composition, t-2 (month-start timing) | 95.4% | 94.8% | 95.7% | 345 | 16 |
| Five-province pooled composition, t-2 | 33.6% | 98.3% | 1.3% | 345 | 229 |

The apparent local advantage is driven by geographic heterogeneity and persistence of this coarse threshold state. Under this certificate, Shanghai is always wholly above the threshold and hence non-crossing; Fujian remains crossing; Sichuan and Qinghai are mostly wholly below; changes are concentrated in Jilin. The five-province pooled mix remains crossing and therefore cannot represent the distinct local regimes. This is evidence that an aggregate mix can conceal local mean-support states, not evidence of improved hourly CIF prediction or intervals.

### Forward-in-time sensitivity

Fit static province and pooled majority states using only target states through 2022, then evaluate 2023-01 through 2026-09 on the common 145 province-month cases where target, t-2, and all baselines are available. The t-2 local state agrees on 99.3% (144/145) of cases; a province-static state gets 80.0% (116/145), and the five-province static pool gets 60.0% (87/145). The dynamic five-province pooled t-2 state gets 40.0% (58/145). Almost all of the local advantage over the static baseline comes from one Jilin transition. This is a useful time-forward confirmation that recent composition can update a stale province classification, but the small number of transitions and only five deliberately selected provinces make the gain fragile. It is not held-out-region validation.

Shanghai's official hydro and nuclear monthly indicators equal zero in all 81 non-missing months in the 2018-01–2026-09 panel. The other 24 months with missing API values remain unknown/missing; they are not replaced with zero. The user's supplied 2025-09–2026-09 thermal, wind, and solar tables match the API for all 33 overlapping rows, as documented in V392.

## Interpretation and next gate

This result supports using official, time-stamped local generation composition as a **minimal information-budget feature** in a China cold-start CIF benchmark. It does not yet establish the main scientific claim: whether the lagged composition feature reduces hourly CIF error or improves interval coverage at matched width in unseen regions. The current panel has five provinces selected from earlier structure contrasts, and the target state is derived from a simple support threshold. A positive paper claim requires a frozen predictor, genuinely held-out provinces, as-of-valid features, and comparison against composition-free, national-anchor, and standard dynamic baselines using hourly labels and coverage/width metrics.

## Artifacts

- `results/v393-release-lag-identifiability-r1/audit_summary.json`
- `results/v393-release-lag-identifiability-r1/state_agreement_summary.csv`
- `results/v393-release-lag-identifiability-r1/state_agreement_by_province.csv`
- `results/v393-release-lag-identifiability-r1/forward_holdout_summary.csv`
- `results/v393-release-lag-identifiability-r1/release_lag_panel.csv`
- Analysis: `scripts/analyze_v393_release_lag_identifiability.py`

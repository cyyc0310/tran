# V6 coarse-risk controlled mechanism checks

All observations in this experiment are generated artificial data. No real target hourly labels, external evaluation files, or external scores are read. These results diagnose mechanisms and limitations; they are not evidence of external generalization. The frozen V6 module, existing tests, and real-data protocol were not changed.

## Reproduction and fixed design

Run from the repository root:

```bash
PYTHONPATH=src ../tran-stable/.venv/bin/python scripts/simulate_coarse_risk_v6.py
```

The script generates all data in memory and writes `results/coarse-risk-v6-controlled/{experiment.json,metrics.csv,summary.csv}`. The JSON records script/core hashes, feature ranges/support, selections, and exact twin identity assertions. Existing results are checked against a repeated run to tolerance `1e-12`; the diagnostics-only extension reproduced the initial metrics without parameter changes.

- Three fixed seeds: 17023, 17024, 17025. Three scenarios, five methods, 45 total fits.
- Two source domains S1/S2 and two artificial unseen domains T1/T2.
- Monthly gas/wind shares for January 2022–December 2026 are finite simplex rows. Gas has emission factor 400; wind has zero. Native publication is the next month start plus 30 days. Feature construction uses the existing release-matched API without additional delay.
- Every method fits the same source 2024 H1 rows and uses only source 2024 H2 for its paired residual bank. Each half contains 216 rows: 18 origins/month × six months × two domains.
- Source 2025 supplies 72 selection rows. Only the interval multiplier is selected, from `{0.8, 1.0, 1.2}`, by mean 90% interval score. There is no method-specific tuning or target selection.
- Artificial unseen 2026 H1 supplies 18 origins/domain, on days 3, 13, and 23 of each month. All tables average the same three seeds; non-twin tables additionally average the two domains equally.
- All methods receive the same 17 features and constant point-model component inputs: level scale 10, shape scale 3. The five frozen methods are global, loglinear, extratrees, structured, and structured_nohindcast. The last is the explicit signal ablation.
- The common point forecast is the latest published same-calendar-month coarse level plus a fixed amplitude-6 daily sine. Every proper/selection/evaluation date has an eligible prior-year same-month observation. Fine mean noise has standard deviation 2; hourly shape noise has standard deviation 1 before removal of its daily mean.

Intervals are deterministic full-bank marginal intervals. Ensemble samples are not used to choose or score interval bounds. The 90% interval score is mean width plus `20 ×` either outside-bound distance. Lower is better. `level_mae` in the CSV evaluates the corrected point's daily mean; the residual-bank distribution can have an additional nonzero mean. Hourly observations within a day and across these small synthetic cohorts are dependent; the three seeds do not justify significance or coverage-guarantee claims.

## 1. Large repeatable season, small innovations

Monthly level is `200 + A sin(2π month/12) + ε_month`, where `ε_month ~ N(0,1)` is shared across domains within a seed. Source amplitudes are 60 and 100; unseen amplitudes are 130 and 160. Fine-level and coarse-level relations are unchanged across domains. Source shape amplitudes are 4 and 8; both unseen domains use amplitude 8.

Unseen raw monthly SD ranges from **91.93 to 113.71**, while seasonal hindcast RMS ranges from **1.34 to 2.09**. Every unseen seasonal hindcast has 24 genuinely valid months. Thus large marginal variation and small unpredictable coarse innovation coexist in lawful inputs. The harmonic approximation is imperfect: its centered error SD is 5.55–7.71. Neither statistic is forced to zero.

| Method | Coverage90 | Width90 | Interval score90 |
|---|---:|---:|---:|
| global | 89.8% | 11.55 | 14.36 |
| loglinear | 98.0% | 27.70 | 28.64 |
| extratrees | 87.7% | 11.19 | 14.88 |
| structured | 87.3% | 11.80 | 15.35 |
| structured_nohindcast | 81.8% | 12.94 | 18.94 |

The coarse features distinguish predictable seasonal amplitude from innovation. This does **not** establish a performance advantage for structured transport: global has slightly better aggregate interval score here, and all methods remain sensitive to finite source calibration. The wide loglinear result is a useful negative comparator. No V5 uncertainty candidate was fitted in this check; raw SD is a mechanism diagnostic, not a reported V5 interval result.

## 2. Directional coarse drift with an unchanged fine-level mapping

Monthly level is `100 + 20 sin(2π month/12) + b t + ε_month`, with month index `t` starting at zero in January 2022. Source slopes are `b=1,2`; unseen slopes are `b=2.5,3`. **Only the observable coarse drift changes. The coarse-to-fine mapping is identical in every domain.** Fine daily mean equals the current month's coarse level plus independent zero-mean noise. There is no hidden fine-level shift.

Because the common point uses last year's same month, the exact level residual is

```text
fine_mean - point_mean = 12 b + ε_current - ε_same_month_last_year + fine_mean_noise.
```

Expected source residuals are 12 and 24; expected unseen residuals are 30 and 36. The signed seasonal hindcast bias estimates this same quantity using previously released monthly outcomes. For seed 17023, source-fit residual means are 12.04/24.11 and coarse bias means are 12.16/24.16. Source-2025 residual means are 12.89/24.42 and coarse bias means are 12.16/24.16. Unseen residual means are 28.39/34.16 and coarse bias means are 30.10/36.10. Across seeds, unseen seasonal bias ranges 30.01–36.33; its centered SD remains only 1.34–2.09.

| Method | Coverage90 | Width90 | Interval score90 |
|---|---:|---:|---:|
| global | 11.3% | 18.62 | 123.75 |
| loglinear | 80.6% | 30.91 | 37.36 |
| extratrees | 24.3% | 11.02 | 85.40 |
| structured | 36.0% | 13.38 | 54.17 |
| structured_nohindcast | 85.9% | 11.81 | 15.39 |

The full structured mechanism fails this extrapolation relative to its no-hindcast ablation, even though it improves over global. This failure is **not** an information-theoretic impossibility: the signed coarse signal is available, meaningful, and obeys the same mapping in source and unseen domains.

The aggregate corrected-location/scale diagnostics identify the observed failure more precisely:

| Method | T1 predicted bias | T2 predicted bias | T1 level scale | T2 level scale |
|---|---:|---:|---:|---:|
| global | 0.00 | 0.00 | 18.84 | 18.84 |
| loglinear | 32.82 | 40.02 | 7.83 | 8.30 |
| extratrees | 23.37 | 23.37 | 1.81 | 1.81 |
| structured | 37.04 | 44.36 | 2.93 | 3.14 |
| structured_nohindcast | 25.18 | 31.77 | 2.24 | 2.29 |

Actual unseen mean residuals average **29.52 and 35.48**. Structured bias therefore overcorrects by about 7.52/8.88 while its conditional level scales remain around 3. The resulting full-bank widths do not absorb this location error. ExtraTrees instead undercorrects beyond the source slope support. The no-hindcast model's corrected point also retains error; its held-out residual bank contributes to its stronger interval result. These observations establish poor learned extrapolation/interval adequacy in this construction; they do not isolate a unique causal culprit among ridge shrinkage, correlated signed signals, or source-support geometry. No parameters were adjusted after seeing this outcome.

## 3. Identical coarse history, unseen fine-shape amplification

All domains have the same coarse law `200 + 30 sin(2π month/12) + ε_month`. Source fine-shape amplitudes remain 4/8. T1 uses amplitude 8 plus zero-mean hourly noise; T2 multiplies **that exact same daily fine shape, including its zero-mean noise, by four**. Daily truth levels remain equal. This is deliberately an unobserved fine-shape shift, unlike the visible coarse slope change in scenario 2.

The script verifies, for all 15 seed/method combinations:

1. T1/T2 monthly rows are strictly equal after removing the domain label.
2. Ordered features and point inputs are strictly equal.
3. Every returned prediction array is strictly equal, including ensembles, interval bounds, component scales, corrected point, and bias.
4. True daily means agree to `1e-12`; only the fine shape differs.

| Method | T1 coverage90 | T2 amplified coverage90 | Shared width90 |
|---|---:|---:|---:|
| global | 86.7% | 12.2% | 11.55 |
| loglinear | 96.3% | 29.4% | 21.99 |
| extratrees | 84.4% | 12.1% | 11.50 |
| structured | 85.9% | 12.3% | 11.67 |
| structured_nohindcast | 77.2% | 15.9% | 12.98 |

Structured's shared width is 11.67 even though coverage falls from 85.9% to 12.3%. Its seasonal coarse RMS is still just 1.34–2.09. Good coarse predictability cannot identify this fine-shape change. Any predictor restricted to these identical observations and random draws must produce the same output in the two worlds. Wider unconditional uncertainty could cover more of both, at a width cost, but no observed coarse signal tells the model which world applies.

## Implications and boundaries

The experiments support separating predictable seasonal variation, signed coarse drift, and unobserved shape uncertainty as distinct quantities. They also supply two concrete limitations: the frozen structured estimator does not reliably extrapolate even a valid signed relationship, and coarse-only conditioning cannot detect an arbitrary fine-shape amplification that leaves its inputs unchanged. The synthetic construction contains only two source domains, one point method, fixed publication delay, three seeds, and a small unseen calendar sample. It neither replaces the real source comparison nor changes any decision in the frozen external evaluation.

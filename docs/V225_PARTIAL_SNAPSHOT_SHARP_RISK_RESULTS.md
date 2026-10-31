# V225 Monthly margins identify a four-field risk frontier

Date: 2026-10-30. Final development run: `results/v225-partial-snapshot-sharp-risk-r3/`.

## Main field frontier

On the 24-region, 135-region-month EIA development cohort, 12 random local days gave:

| Fields observed per hour of sampled days | Macro hourly coverage | Simultaneous 24-hour day coverage | Width | Winkler-90 |
|---|---:|---:|---:|---:|
| Exact 8 fuels | 99.05% | 94.32% | 221.74 | 225.22 |
| Total + thermal + coal + gas | 99.10% | 94.52% | 223.34 | 226.64 |
| Total + thermal + coal | 99.25% | 95.50% | 259.88 | 262.45 |
| Total + thermal | 99.46% | 96.92% | 377.35 | 378.91 |
| Total + coal | 99.98% | 99.82% | 516.84 | 517.07 |
| Total only | 100.00% | 100.00% | 645.68 | 645.68 |

The four-field interval was close to exact-eight-field calibration: width difference +1.60 gCO2/kWh [region bootstrap 95% CI +0.02,+3.72], Winkler difference +1.41 [+0.01,+3.34], and simultaneous day coverage difference +0.20 percentage points [+0.004,+0.52]. All 24 regions had average simultaneous-day coverage above 90%; the worst was NWMT at 91.82%.

The three-field total+thermal+coal combination used one fewer local series, kept every region above 90% average day coverage (worst EPE 92.02%), and widened intervals by 38.13 [11.10,69.09] with Winkler increase 37.22 [11.95,68.36] relative to exact eight fields. Total+thermal alone also cleared 90% in every region but had much larger intervals. Total+coal and total-only were too conservative to be efficient.

For context, the source-only daily-max baseline at 12 days had width 267.75, Winkler 295.88, macro hourly coverage 95.65%, and simultaneous day coverage 83.60%. Full-eight local and four-field local calibration both improved macro Winkler while raising day coverage; the four-field version's average width (223.34) was also below the source baseline.

At 9 days, the four-field method was already close to full eight fields in macro metrics, but eight regions fell below 90% average day coverage. Twelve days is the smallest tested budget that removed all below-90% region averages for both the four-field and exact-eight-field local methods.

## What is being identified

The monthly margins plus hourly total and thermal aggregate pin down the daily fossil-generation amount. Adding hourly coal and gas measurements then constrains the remaining oil/other fossil split. This sharply reduces the feasible hourly EF range. In this EIA proxy, four measured hourly channels per time point were sufficient to approximate the exact-eight-channel risk interval; reducing to three channels retained coverage but cost about 17% width.

This is a risk-identification result, not a point-prediction gain. It depends on knowing all eight target-month fuel totals and on collecting 12 full days of hourly aggregates after the month. It cannot be run from the user's monthly tables alone.

## Method novelty boundary

The combination of feasible-set score upper bounds and conformal calibration is related to work on conformal prediction with partially labelled data and robust calibration with upper-bounded scores. The potentially new part is the power-system feasible set induced by fuel-specific monthly conservation margins, the day-maximum hourly-CIF risk target, and the measured-field value frontier. A generic conformal-method novelty claim is not supported. See [conformal prediction with partially labeled data](https://proceedings.mlr.press/v204/javanmardi23a.html) and [verifiably robust conformal prediction](https://papers.neurips.cc/paper_files/paper/2024/file/0814a342597d65e0832fc7ec9b42c317-Paper-Conference.pdf).

## Files

- [Frozen V226 confirmation protocol](V226_FOUR_FIELD_PARTIAL_RISK_FROZEN_PROTOCOL.md)
- [Development runner](../scripts/v225_partial_snapshot_sharp_risk_bounds.py)
- [V225 protocol](V225_PARTIAL_SNAPSHOT_SHARP_RISK_PROTOCOL.md)

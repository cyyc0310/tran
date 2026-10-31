# V131 — Daily Risk-Side Summary Adds Temporal Information

## Result

The frozen V129 cohort contains 358 region-months across 21 EIA balancing areas (2,148 region-month × threshold × target-rate rows). All inclusion checks passed: 0 nesting failures. All four information contracts were scored on the same observed hourly records. No hourly source rows or regions were added after seeing outcomes.

Equal-region mean interval widths (hours) are:

| Threshold | Monthly risk-side | Daily risk-side | Daily detailed-fuel oracle | Monthly detailed-fuel oracle | Monthly-to-daily shrinkage |
|---:|---:|---:|---:|---:|---:|
| τ=600 | 316.08 | 225.16 | 222.28 | 315.97 | 90.92 h (28.8%) |
| τ=900 | 213.50 | 192.76 | 189.42 | 213.38 | 20.73 h (9.7%) |

The daily risk-side interval is only 2.88 h wider than the daily detailed-fuel oracle at τ=600 and 3.34 h wider at τ=900 (about 1.3% and 1.8% of the respective oracle widths). Thus, under this EIA proxy and factor-support stress, temporal partitioning carries most of the observed additional identification value; revealing daily fuel-by-fuel splits adds little beyond the daily risk-side statistic.

At the ≥1% count decision, τ=600 abstention falls from 43.99% with monthly risk-side information to 43.28% with daily risk-side information. For τ=900 it stays at 29.65%. At τ=600, daily risk-side information issues a positive certificate in 0.71% of region-months; the monthly contract issues none. This is a small but measurable decision gain, not a broad high-tail detection result. Daily information also makes the count interval no wider than its monthly counterpart in any of the 21 regions in this cohort; width falls in 11 regions at τ=600 and 7 at τ=900, and is equal in the others.

## What the information contract means

The daily contract releases three UTC-day quantities: thermal generation total, generation on the factor-support side above the chosen threshold, and clean generation total. The comparison holds technology capacity out of every arm. For τ=600, the threshold-side quantity is coal+oil MWh; for τ=900 it is coal MWh, with the τ=900 low-factor endpoint retaining K=0. Daily fuel totals are only used for the oracle comparator.

## Research interpretation

This is evidence for a **threshold-specific temporal information budget**: monthly composition alone cannot resolve many hourly count decisions, while three daily aggregates can recover almost all the interval tightening obtained from daily detailed-fuel totals. The result narrows the local-data question from “publish every fuel hourly” to testing whether a small daily reporting contract is sufficient.

It is not yet a new CIF prediction algorithm, a calibrated risk forecast, or proof of Chinese operational value. The cohort is retrospective and selected by the V90 capacity audit, the underlying hourly emissions labels are fixed-factor EIA proxies, and the EF intervals are stress assumptions. Daily reporting vintage, source error, regional transfer, actual Chinese provincial data definitions, and real hourly CIF labels remain unvalidated. The daily interval is built by summing robust daywise bounds, which is conservative because it does not exploit common-factor dependence across days.

## Reproduction

- Protocol: [V131 protocol](V131_DAILY_RISK_SIDE_INFORMATION_PROTOCOL.md)
- Script: [V131 runner](../scripts/v131_daily_risk_side_information.py)
- Outputs: `results/v131-daily-risk-side-information-r1/`
- Protocol SHA-256: `b91e60713db17aaf1c59b4c41e7a41d99fb35ee799af5385f47dbc786571baa9`
- Script SHA-256: `24fc73dd41f6b7f0e036ff040928bb0c04728d124f40a40a71c1cb84b5738cdd`
- Summary SHA-256: `d1824539582b55a64159871a32ed3e2e75fad52d88417ac2dfbfcc9834c5f406`

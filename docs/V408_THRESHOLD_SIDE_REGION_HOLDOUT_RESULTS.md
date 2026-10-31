# V408: threshold-side monthly energy rule on held-out system regions

Date: 2026-10-31. This is a roster-held-out replication of the V97/V98 risk-side information rule, evaluated on EIA-930 generated fixed-EF proxy labels. The frozen protocol is [V408](V408_THRESHOLD_SIDE_REGION_HOLDOUT_PROTOCOL.md); machine-readable outputs are in [`results/v408-threshold-side-region-holdout-r1/`](../results/v408-threshold-side-region-holdout-r1/).

## Result

The frozen roster contains all 41 V52 quality-census regions outside the V90/V97 development roster. In 2025, 29 regions supplied 247 region-months with at least 80% valid hourly proxy rows. No months were selected using threshold counts or capacity agreement. Lagged 2024 EIA-860 coarse capacity passed the fixed feasibility check for 140/247 months (56.7% pooled; 58.9% region-equal); every other month remains in the denominator and capacity-based decisions abstain.

At both tested thresholds, the compact threshold-side field retained nearly all of the hourly-count set information from detailed monthly fuel margins:

| Threshold | Risk-side field | No-capacity mean set width: side / detailed | Exact width matches | Capacity-usable width: side / detailed | Exact matches among usable months |
|---|---|---:|---:|---:|---:|
| 600 gCO₂/kWh | Coal + oil MWh | 480.36 / 480.34 h | 239/247; all remaining gaps ≤1 h | 305.10 / 299.68 h | 123/140; maximum gap 56 h |
| 900 gCO₂/kWh | Coal MWh | 453.58 / 453.55 h | 236/247; all remaining gaps ≤1 h | 212.20 / 209.10 h | 127/140; maximum gap 65 h |

For every eligible region-month, the detailed-margin count set was contained in the threshold-side set, and the observed proxy count was contained in both. The same containment held on capacity-usable rows after adding the identical lagged coarse-capacity envelope to each information set. Thus the side field is a strong *compression* of detailed monthly fuel reporting for this fixed-EF count-set construction, with a small average width penalty on usable rows (5.43 h at 600; 3.10 h at 900) and material outliers that remain visible in the rowwise audit.

The operational frontier remains weak. Without capacity, region-equal positive/negative/abstention rates at the 1% event-rate target were 3.45/29.02/67.53% at 600 and 3.45/32.76/63.79% at 900. With capacity required, unavailable-capacity rows counted as abstentions; at the 1% target, the all-eligible side-rule frontier was 0.29/20.38/79.34% at 600 and 0/22.39/77.61% at 900. Changing the event-rate target to 5% or 10% changes the frontier but not set widths; the full grid is in `summary.json` and `macro_frontier.csv`.

## Interpretation and limits

This is the strongest current evidence for a research contribution around **information-budgeted partial identification**: identify the minimum threshold-specific monthly fuel aggregate needed to preserve an auditable hourly tail-count set, rather than proposing another generic forecasting network. The result replicates outside the mechanism-development roster and makes the cost of using a coarser field measurable.

It is not an hourly forecast, a probabilistic coverage result, a direct emissions measurement, or China evidence. Labels are constructed from EIA-930 generation using fixed emission factors; the side aggregate is treated as exact; capacity is feasible for only 56.7% of eligible months; and the held-out roster is not a project-wide blind test. The positive-decision rate is low and abstention high, so near-sufficiency must not be described as high practical forecast accuracy.

The user's Shanghai NBS files do not contain the required coal+oil / coal monthly split. Their aggregate thermal generation and zero hydro/nuclear records can inform a broad monthly mean-support state, but do not identify subfuel composition, hourly dispatch, or the tail-count set. This mechanism therefore remains a **measurement-design candidate** for China, not a currently deployable Shanghai hourly predictor.

## Reproduction and audit

The analysis reuses the deterministic V97 set construction and the frozen V408 cohort. Exact input and helper-script hashes, regional capacity audit, containment checks, and row-level widths are preserved in `summary.json`, `roster_and_capacity_audit.csv`, `region_macro_frontier.csv`, and `heldout_region_month_frontier.csv`. Aggregation uses region-equal averages; capacity-conditioned widths use only feasible rows, while all-roster decision rates count infeasible capacity as abstention.

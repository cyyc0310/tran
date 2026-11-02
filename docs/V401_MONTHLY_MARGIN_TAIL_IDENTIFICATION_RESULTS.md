# V401 results: monthly margins leave wide hourly tail-count ambiguity

Date: 2026-11-02. Frozen protocol: [V401](V401_MONTHLY_MARGIN_TAIL_IDENTIFICATION_PROTOCOL.md). Runner: `scripts/v401_monthly_margin_tail_identification.py`. Tracked machine evidence: [V401 evidence package](evidence/v401-monthly-margin-tail-identification-r2/README.md).

## Design

This is a generation-side information-identification experiment, not a CIF forecasting evaluation. It uses 2024 EIA hourly data from seven balancing authorities, aggregates coal/gas/oil into a China-compatible monthly thermal category, and defines a deliberately simple fixed-factor proxy as `700 × hourly thermal share`. Hydro, nuclear, wind, solar and EIA's separate `other` residual have zero proxy intensity. The response is not measured emissions, dispatch-aware marginal CIF, or consumption-side CIF.

The information ladder is monthly five-class generation margins (**M**), the same margins plus the true hourly total generation profile (**M+T**, an oracle), and M+T plus the true UTC-day thermal generation margins (**M+T+D**, another oracle). M has an analytical sharp count bound using the monthly energy-weighted proxy mean. MILPs give feasible count intervals under M+T and M+T+D while hiding the hourly thermal allocation. Neither oracle is in the currently stated China data contract. We score thresholds 300/500/700 g/kWh. The 700 threshold is structurally impossible to exceed under the fixed 700 thermal factor, so it is only a boundary control.

## Results

Region-equal mean interval widths across 77 valid region-months:

| Threshold | M-only width | M+T width | M+T+D width | M+T+D incremental narrowing vs M+T | Largest regional M+T+D width |
|---:|---:|---:|---:|---:|---:|
| 300 g/kWh | 700.0 h (99.8% month) | 492.1 h (70.3%) | 463.8 h (66.3%) | 5.6% | 83.8% (CISO) |
| 500 g/kWh | 700.0 h (99.8% month) | 493.6 h (70.9%) | 468.0 h (67.3%) | 5.1% | 87.0% (FPL) |
| 700 g/kWh | 0 h | 0 h | 0 h | Not applicable | 0% |

M+T+D narrows the M-only set by 33.6% at 300 and 32.6% at 500, but adds only 5.6% and 5.1% narrowing over M+T. The observed proxy count lies inside all feasible bounds in every cell, and all MILPs have zero reported optimality gap. That 100% inclusion is a feasibility check: observed hourly allocations generated the monthly, daily, and hourly totals used in the optimization. It is not a forecast-coverage result.

MISO remains especially ambiguous at 500 g/kWh. Its mean M+T interval is approximately 8.5–671.7 hours (663.2 hours wide, 91.1% of a month); adding daily thermal margins changes it to approximately 24.5–652.0 hours (627.5 hours wide, 86.2%). The observed proxy averages 197.4 hours above threshold. At 300 g/kWh the MISO M+T+D interval averages 277–728 hours (61.9% width). Coarse daily thermal energy only modestly narrows the hourly-total oracle set.

Valid-month support is uneven: BPAT, CISO, FPL, MISO and NYIS have 12 months; ERCO has 9 and ISNE 8 after the common hourly validity rule. All comparisons use the months available within each region and then weight regions equally.

## Decision

The results support a narrower conclusion: hourly total generation reduces the M-only set by about 29%, while daily thermal margins reduce the M+T set by only another 5%. The final set still spans 66%–67% of a month on average and 86% in MISO at the 500 threshold. A credible hourly generation-CIF method needs finer local temporal information about fuel dispatch or emissions factors, or it must return broad bounds or abstain. Current evidence does not show that foreign pseudo-target learning can recover the missing information.

The result pertains only to the generation-side proxy. It cannot be transferred to the consumption-side track: monthly local generation margins do not provide cross-region transfer flows, source-region intensity, or the destination accounting rule.

The 2024 regions were previously opened, so this is development evidence only. No target country was frozen or tested as an unseen confirmation.

## Reproducibility and audit

The r7 machine summary records source-data and runner SHA-256 values. It contains 231 region-month-threshold rows (77 × 3), with no inverted bounds, no observed proxy count outside any bound, and zero nonzero solver gaps. Earlier scratch outputs are not evidence: some used a broad M-only bound, all-month rather than region-equal aggregation, merged EIA's `other` residual into solar, or missed the structurally zero threshold control. The r7 runner reproduces M+T and M+T+D together. Re-run with:

```bash
python scripts/v401_monthly_margin_tail_identification.py
```

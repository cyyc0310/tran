# V50 monthly activity scale adds no reliable transfer signal

Run date: 2026-10-27. Frozen design: [V50_MONTHLY_ACTIVITY_SCALE_PROTOCOL.md](V50_MONTHLY_ACTIVITY_SCALE_PROTOCOL.md). The same 70 external prediction packages and ten label packages were used; no target or unseen source was opened, and no point/risk model was refit.

## Source-macro results

| Daily correction feature | Hourly CRPS | Daily level MAE | 90% coverage | Width | Interval score | Clipping |
|---|---:|---:|---:|---:|---:|---:|
| None (raw anchor) | 125.14 | 144.61 | 88.50% | 527.98 | 712.16 | 5.66% |
| Equal-source global bias | 112.80 | 141.69 | 84.83% | 518.36 | 784.46 | 0.11% |
| Fuel-mix EF proxy | 71.43 | 93.47 | 95.27% | 478.64 | 510.42 | 10.05% |
| Monthly activity scale only | 155.79 | 185.42 | 75.25% | 483.46 | 1407.56 | 4.25% |
| Fuel mix + activity scale | 92.95 | 120.18 | 88.64% | 466.95 | 647.56 | 9.84% |

The activity feature was the mean over the three latest legal monthly contexts of `log1p(total_mwh / expected_hours)`. Its standalone correction is substantially worse than both the raw anchor and fuel-mix correction. Adding it to fuel mix narrows the interval, but loses coverage and worsens CRPS/interval score relative to fuel mix alone. It is therefore not a useful generic extra feature in this source set.

The source pattern is unstable: activity-only correction has zero coverage at SCL and SWPP, while SEC improves relative to the fuel-mix arm; the combined model collapses SEC coverage to 72.2% and SWPP to 36.1%. Total monthly output appears to encode source scale/identity effects that do not transfer as a stable daily bias rule.

## Decision

This does not support adding monthly generation volume as a universal level-correction feature. The fuel-mix proxy remains the only tested monthly field with a clear pooled signal, but unresolved SC extrapolation, SEC undercoverage, SCL clipping, and source heterogeneity prevent freezing it. The tested “minimum useful information” is therefore not simply total monthly generation: composition matters more in this proxy experiment. We still cannot infer which local data will make hourly tails identifiable or claim China transfer.

## Artifacts

- Script: `scripts/probe_v50_monthly_activity_scale.py`
- Output: `results/disjoint-calibration-v50-monthly-activity-scale/`
- The run summary binds the protocol, script, frozen seals, monthly input hashes, and result hashes.

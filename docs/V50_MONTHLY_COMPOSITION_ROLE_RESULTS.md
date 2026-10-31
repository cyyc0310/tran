# V50 monthly composition roles: low-dimensional source-held-out ablation

Run date: 2026-10-27. Frozen protocol: [V50_MONTHLY_COMPOSITION_ROLE_PROTOCOL.md](V50_MONTHLY_COMPOSITION_ROLE_PROTOCOL.md). All model fits were outer source-held-out; no target/unseen source was opened.

## Macro results

| Daily correction feature | Hourly CRPS | Daily level MAE | 90% coverage | Width | Interval score | Clipping |
|---|---:|---:|---:|---:|---:|---:|
| Raw anchor | 125.14 | 144.61 | 88.50% | 527.98 | 712.16 | 5.66% |
| EF proxy | 71.43 | 93.47 | 95.27% | 478.64 | 510.42 | 10.05% |
| EF + fossil share | 64.53 | 72.38 | 86.34% | 483.74 | 628.55 | 2.06% |
| EF + variable renewables | 88.40 | 109.84 | 90.28% | 472.72 | 677.53 | 11.07% |
| EF + fossil + variable renewables | 81.35 | 92.37 | 81.08% | 477.28 | 759.48 | 2.26% |
| Grouped composition only | 156.66 | 184.74 | 73.44% | 434.90 | 1559.82 | 21.02% |

Fossil share adds source-held-out level information beyond the fixed-EF scalar: its average CRPS and daily level MAE improve. However it is not a calibrated interval correction. SC coverage falls to 61.2%, and macro coverage is only 86.3%; interval score also worsens. The variable-renewable term is less useful, while the full grouped-composition regression is unstable and clipping-heavy. These failures are not cured by adding more unregularized composition terms.

## Source pattern

EF + fossil share materially reduces CRPS for NWMT (160.6→104.2), SEC (120.4→44.1), and WACM (124.2→63.6), but SC deteriorates (74.1→112.1 CRPS; 99.9%→61.2% coverage), and SCL's zero-fossil mix makes the learned source transfer badly. This is consistent with a potentially useful fossil dispatch-role feature, but also with a brittle cross-region slope.

## Decision

The result motivates a fixed shrinkage-path diagnostic for the incremental fossil-share coefficient. It does not justify a new method claim: the OLS split fails the coverage gate and includes the same ten previously used sources. No coefficient or feature set is frozen; no unknown source has been accessed.

## Artifacts

- Script: `scripts/probe_v50_monthly_composition_role.py`
- Output: `results/disjoint-calibration-v50-monthly-composition-role/`
- Run summary binds the protocol, script, data seals, monthly inputs, and outputs.

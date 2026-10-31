# V50 fossil-role coefficient shrinkage path

Run date: 2026-10-27. Fixed protocol: [V50_FOSSIL_ROLE_SHRINKAGE_PROTOCOL.md](V50_FOSSIL_ROLE_SHRINKAGE_PROTOCOL.md). This is a fixed-grid descriptive development path; no λ is independently confirmed.

| Fossil-role multiplier | Hourly CRPS | Level MAE | 90% coverage | Width | Interval score | Clipping |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (EF only) | 71.43 | 93.47 | 95.27% | 478.64 | 510.42 | 10.05% |
| 0.25 | 63.35 | 83.37 | 96.14% | 482.64 | 500.15 | 6.02% |
| 0.50 | 59.74 | 76.58 | 94.92% | 485.31 | 511.46 | 2.51% |
| 0.75 | 60.45 | 72.42 | 89.84% | 485.39 | 556.24 | 1.80% |
| 1.00 | 64.53 | 72.38 | 86.34% | 483.74 | 628.55 | 2.06% |

At λ=.25, mean CRPS and interval score improve slightly against EF-only, while coverage is higher; but width increases by 4.0 (0.84%). λ=.50 improves CRPS but worsens interval score and increases width. Larger λ violates the coverage gate. The source-macro minimum at λ=.25 is SCL at 85.98%, close to the 85% floor; its particle clipping remains 49.0% (raw anchor 56.5%). These are promising development metrics, but λ=.25 was selected after examining this same fixed path and requires nested validation.

## Decision

Fossil share has incremental level-prediction information, but this fixed λ path does not establish a robust efficiency improvement over the EF-only baseline. It motivates the nested coefficient-selection test; it does not trigger method freeze or unseen-source access.

## Artifacts

- Script: `scripts/probe_v50_fossil_role_shrinkage.py`
- Output: `results/disjoint-calibration-v50-fossil-role-shrinkage/`

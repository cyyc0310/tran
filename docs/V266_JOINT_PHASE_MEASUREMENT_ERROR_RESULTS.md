# V266 results: joint phase and measurement error leaves most temporal information value

Date: 2026-10-30. This reuses the already-open V264 EIA proxy cohort; it is not a new-region or China validation.

## Result

We jointly allowed a one-hour within-day phase ambiguity and multiplicative error in the hourly total generation curve, while keeping daily totals and monthly fuel margins exact. The V264 phase-only reference supplied 12 fully proved BA-months. In this follow-up, 10/12 cases had both endpoints proven under both noise levels; 44/48 endpoints were optimal. The two incomplete cases were PSCO April and SOCO October. Their incumbents are retained in the raw table but excluded from the paired sharp-bound averages.

| Hourly-total information | Complete cases | Mean phase-only width | Mean joint width | Mean daily-only width | Extra width vs phase-only | Reduction vs daily-only |
|---|---:|---:|---:|---:|---:|---:|
| Phase envelope + ±5% error | 10 | 627.3 h | 633.1 h | 730.9 h | 5.8 h | 97.8 h (13.4%) |
| Phase envelope + ±10% error | 10 | 627.3 h | 637.8 h | 730.9 h | 10.5 h | 93.1 h (12.7%) |

On the matched 10 cases, the daily-to-phase-only reduction is 103.6 h. Joint phase and measurement uncertainty retains 94.4% of this reduction at ±5% and 89.9% at ±10%. All 20 proved sets contain the observed fixed-EF proxy tail count and nest inside the daily-only set. PACE still has much weaker information gain: 20–41 h shrinkage across its three months, compared with 97–215 h in the other fully proved BA-months.

## Interpretation

The V264 temporal-information signal does not vanish when modest multiplicative measurement error is added to the phase envelope. But this is a secondary robustness result, not a new contribution: the gains are set-width reductions in a fixed-EF EIA activity proxy, and the exact daily totals are still given. It does not show predictive coverage or provide evidence that Chinese monthly tables alone identify hourly risk.

The practical question remains what high-frequency side information can actually be obtained in a target province. The user's confirmed monthly all-source generation data can anchor monthly composition/production-side intensity under compatible factors and boundaries. It does not provide the hourly-total curve used here. The present China evidence also does not yet include the original monthly source file, its schema, first-publication dates, or revision history.

## Reproducibility

- Protocol: `docs/V266_JOINT_PHASE_AND_MEASUREMENT_ERROR_PROTOCOL.md`
- Runner: `scripts/v266_joint_phase_measurement_error.py`
- Endpoints, case bounds, summary, and hashes: `results/v266-joint-phase-measurement-error-r1/`


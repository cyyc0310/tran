# V129 — Factor-Support-Aware Risk-Side Monthly Information

## Result

On the frozen 358 EIA region-month cohort, the threshold-indexed risk-side summaries nearly reproduce the detailed-fuel robust K endpoints under fuel-specific EF intervals, and remain much tighter than the public-style thermal-total budget. All 2,148 scored threshold/rate rows contain the nominal proxy K; there are zero capacity-feasibility failures and zero interval-nesting failures.

The table reports the project’s region-equal macro aggregation, matching V96/V97. EF intervals are coal [900,1100], gas [360,440], oil [720,880], other fossil [200,600] gCO₂/kWh, and 0 for nonthermal sources.

| τ | Monthly information + same coarse capacity | Mean K width (h) | Positive / negative / abstain at ≥1% |
|---:|---|---:|---:|
| 600 | Detailed fuel margins | 222.1 | 4.5% / 56.0% / 39.5% |
| 600 | Risk-side coal+oil MWh | 232.6 | 0.5% / 56.0% / 43.5% |
| 600 | Thermal total only | 563.2 | 0% / 19.0% / 81.0% |
| 900 | Detailed fuel margins | 148.6 | 0% / 70.3% / 29.7% |
| 900 | Risk-side coal MWh, with coal EF crossing τ | 162.4 | 0% / 70.3% / 29.7% |
| 900 | Thermal total only | 483.3 | 0% / 19.0% / 81.0% |

Without capacity, the risk-side mean width differs from the detailed-fuel robust set by only 0.10 h at τ=600 and 0.11 h at τ=900. With the same coarse capacity, it is 10.4 h wider than detailed fuel margins at τ=600 and 13.8 h wider at τ=900, while reducing width by 58.7% and 66.4%, respectively, versus thermal-total-plus-capacity. At τ=900 the one-field risk-side and detailed arms make the same negative/abstain decisions at all three risk targets; neither certifies positive tail risk under this wide factor support.

## Why this is a stronger candidate

V96/V97 showed that threshold-side monthly MWh nearly reproduces detailed margins under narrower EF ranges. V129 shows this is not merely an artifact of fixing coal above τ=900: with coal EF spanning [900,1100], the low endpoint allows no strict exceedance, while the high endpoint yields the largest coal-side feasible set. The interval hull remains close to the full detailed-margin envelope. At τ=600, coal+oil is definitely above the threshold, while the remaining thermal group can reach EF=τ but cannot exceed it; the solver handles that neutral boundary under the strict event definition.

This suggests a **factor-support-aware, threshold-indexed information compression**: keep one monthly MWh aggregate for the risk-side/threshold-crossing fuels, retain other fuel groups as interval-valued, then combine with coarse technology capacity to return a robust K-set. The minimal local field changes with τ and with which factor intervals cross the threshold. This is now a stronger paper mechanism candidate than the V121/V122 support-gap fallback because it produces an interpretable data-sufficiency result against detailed-margin and coarse-capacity baselines.

## Limits

The risk-side fields are oracle aggregates from U.S. EIA hourly fuel data, not a published Chinese field tested prospectively. The 358-month cohort was selected using a retrospective capacity audit; the EF intervals are sensitivity assumptions, not measured Chinese uncertainty. Results are robust partial-identification envelopes, not calibrated forecast intervals or actual Chinese CIF validation. The width reduction is substantial but still leaves 148–163 hours of average uncertainty at τ=900.

## Reproduction

- Frozen protocol: [V129_FACTOR_SUPPORT_RISK_SIDE_PROTOCOL.md](V129_FACTOR_SUPPORT_RISK_SIDE_PROTOCOL.md).
- Runner: [v129_factor_support_risk_side.py](../scripts/v129_factor_support_risk_side.py).
- Paired rows, region-equal frontier, nesting audit and hashes: `../results/v129-factor-support-risk-side-r1/`.

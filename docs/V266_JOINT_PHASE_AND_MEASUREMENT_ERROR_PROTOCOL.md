# V266 protocol: joint hourly-total phase and measurement uncertainty

Date: 2026-10-30. This is a robustness experiment, not a new forecast model.

## Question

Do the tail-count identification gains from hourly generation totals survive when the only available hourly-total curve has both a one-hour phase ambiguity and bounded multiplicative measurement error?

## Frozen design

- Reuse only the 12 complete BA-month cases from V264's already-open 2025 cohort at `tau=500 gCO2/kWh`; do not add or replace cases after inspecting outcomes.
- Keep monthly fuel margins and daily total generation exact, as in V264.
- Shift each true hourly-total profile by one hour within each day. Form the V263 neighbor envelope around that shifted curve, then expand the envelope multiplicatively by ±5% and ±10%.
- Solve exact min/max tail-count MILPs with the same fixed proxy factors and time limit as V264.
- Compare against the same cases' V264 phase-only set and V239 daily-only set. Report the paired interval width and width loss due to joint uncertainty.
- A case counts as confirmed only if both endpoints are optimal, the observed fixed-factor tail count is contained, and the result is nested inside the daily-only set. Incomplete endpoints are reported and excluded from paired means.

## Interpretation limits

The hourly profile and CIF are derived from EIA balancing-area fuel generation with fixed proxy factors. This tests partial-identification robustness, not prediction coverage, actual Chinese measurement quality, or the user's monthly reporting table. The 5%/10% error magnitudes are stress settings, not estimated data-error distributions. The one-hour phase envelope is an artificial clock-uncertainty model.


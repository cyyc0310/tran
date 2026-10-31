# V322 protocol: sensitivity to the unobserved-generation factor support

Registered 2026-10-30 before running the sensitivity extension. Reuse the V321 source inputs, LP, OD matrix, period bins, receipt fields, and information arms without refitting or changing any constraints except the unobserved residual EF upper endpoint.

## Scenarios

- Primary V321 support: `[0, 1.0] kgCO2e/kWh` (already executed and frozen).
- Lower upper-endpoint scenario: `[0, 0.5] kgCO2e/kWh`.
- Higher stress scenario: `[0, 1.5] kgCO2e/kWh`.

The `[0,1.0]` endpoint equals the maximum 2023 Ember source-month factor; 0.5 and 1.5 are sensitivity values, not validated technology-specific bounds. Do not select the best scenario after seeing outcomes.

## Primary reporting

For every scenario, report mean/median diameter in all four information arms, paired province-period relative shrinkage from five categories versus total-only both with and without 310 receipt fields, positive cells, positive provinces, and province-cluster bootstrap 95% intervals. Any nonmonotonicity in maximum width as the EF upper support increases is a calculation or implementation audit trigger.

This remains a structural monthly direct-import exposure proxy; no CIF, hourly coverage, or forecast skill claim is permitted.

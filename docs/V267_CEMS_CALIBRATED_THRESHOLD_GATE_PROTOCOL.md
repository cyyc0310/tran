# V267 protocol: calibrate the threshold-side gate with measured CEMS heterogeneity

Date: 2026-10-30. This is a cross-system stress test of the V265 gate, not a China EF estimate.

## Question

Does the V265 conclusion that fuel-factor precision is a prerequisite to hourly-tail identification survive when the uncertainty scale is calibrated to measured cross-state, within-fuel CEMS heterogeneity instead of a uniform assumed ±10% box?

## Frozen inputs and calculation

1. Use 2025 EPA CEMS records whose CO2 mass measurement indicator is explicitly MEASURED/MEASURE, with positive operating time, gross load, and CO2 mass.
2. Aggregate each state's measured gross generation and CO2 mass by quarter and harmonized group: coal, oil/petroleum coke, and natural gas. Exclude blended/unmapped fuel rows rather than silently reassigning them.
3. For each fuel group and quarter, calculate the generation-weighted national mean factor and the smallest symmetric relative radius around it containing at least 90% of the observed state-quarter generation. This radius is an empirical heterogeneity envelope, not a confidence interval.
4. Transfer those radii only as stress multipliers to the already-open 16 EIA BA-months in V240, applying them to the project's nominal group factors `[1000, 800, 400] g/kWh` and the observed monthly fuel margins. Divide by all-source generation, including non-fossil sources with zero direct EF. Compute the possible monthly mean CIF range and whether it crosses 500 g/kWh.
5. Compare with the V265 uniform ±10% gate. Keep all 16 BA-months; make no new-region inference.

## Interpretation limits

CEMS covers monitored U.S. fossil units and gross load. State-quarter factor dispersion includes plant mix and measurement/operating differences; it is not a target province's EF uncertainty distribution. Transferring its relative radii to EIA balancing areas is deliberately a stress test. A 90%-generation envelope does not guarantee coverage of all generation. The result cannot validate Chinese factors, hourly CIF forecasts, or probabilistic coverage.


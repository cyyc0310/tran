# V268 protocol: separate EF level shift from cross-state dispersion

Date: 2026-10-30. This extends V267 because applying empirical *relative spread* around the project's nominal EF can hide large systematic level differences.

## Question and frozen comparison

Using the same 16 already-open EIA BA-months, compare threshold-side gate results under two distinct CEMS-based stress intervals:

1. **Spread-only:** project nominal factors `[1000, 800, 400] g/kWh`, widened by the V267 quarter-specific 90%-generation state-quarter relative radius.
2. **Level + spread:** center each fuel interval on the measured CEMS national fuel-quarter factor, then widen by the same 90%-generation radius.

The interval for each BA-month is the generation-weighted monthly mean under its observed fuel margins, with the denominator including non-fossil generation (zero direct EF). A case fails the threshold-side gate if its mean-factor interval straddles `500.0001 g/kWh`. Compare the resulting count to V265's uniform ±10% reference.

## Limits

This remains a cross-system transport stress: U.S. CEMS state-quarter factors are not Chinese fuel factors, not a local target uncertainty distribution, and not a confidence interval. CEMS is fossil-unit gross generation; non-fossil is held at zero in the EIA proxy. The gate theorem is conditional on the chosen factor set, not a probability coverage guarantee.


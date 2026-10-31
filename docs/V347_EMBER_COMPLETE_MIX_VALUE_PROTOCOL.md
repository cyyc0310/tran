# V347: value of an estimated complete monthly provincial fuel mix

Registered 2026-10-30 after V346 showed that the NBS above-size annual remainder can return hourly K hulls close to the whole reporting period. This experiment asks how much a publicly available, estimated eight-source monthly mix would narrow the same model-conditional K hull if treated as a complete monthly margin. It does not treat Ember as independent truth.

## Data and target

Use the frozen V294 Ember province-month CSV for 2023, atomic rows only (`Coal`, `Gas`, `Other fossil`, `Bioenergy`, `Hydro`, `Nuclear`, `Wind`, `Solar`). Aggregate January and February to match the NBS/V341 Jan-Feb reporting bin. Restrict scoring to V341's 330 complete province-period keys. Keep the same thresholds, CEC capacity arms and V342 sharp fixed-factor solver; compare against the V346 NBS-plus-residual outer hull.

Collapse Ember's estimated coal, gas, other-fossil and bioenergy monthly generation into the thermal margin. Compute a monthly effective thermal-EF interval by weighting the existing V99 sensitivity bands: coal [900,1100], gas [360,440], other fossil [200,600], bioenergy 0 g/kWh. Hydro, nuclear, wind and solar direct operational EF remain zero. The interval is deterministic and composition-weighted; it is not a probability distribution or measured local factor.

## Comparisons and checks

Report interval widths and the ratio of Ember estimated monthly generation to the NBS above-size total. V341 and V346 result keys must match exactly. Each CEC capacity arm uses the corresponding generation-group capacity vintage; no residue group is added to the Ember arm. At least the 330 × 2 × 3 expected province-period/threshold/capacity rows must be present.

## Interpretation limits

This is an information-value sensitivity: it asks what narrower model sets follow if a secondary estimated, source-corrected monthly mix is accepted as a complete margin. It does not test whether the Ember mix is accurate at province-month scale, whether its estimates were available at a historical forecast origin, or whether the resulting bounds cover Chinese hourly CIF. The NBS and Ember categories may differ in coverage and accounting boundary. The factor bands are proxy sensitivity assumptions. No hourly China labels are used.

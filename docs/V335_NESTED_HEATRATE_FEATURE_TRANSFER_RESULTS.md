# V335 results: local heat-rate information survives nested regularization in selected cold-start comparisons

Run date: 2026-10-30. Frozen folds and choices: [V335 protocol](V335_NESTED_HEATRATE_FEATURE_TRANSFER_PROTOCOL.md). Outputs: `results/v335-nested-heatrate-feature-transfer-r1/`.

## Main comparison

Under seven whole-statistical-region holdouts and forward 2022/2023 targets, five-group mix-only ridge-NNLS is compared with the same-source model that scales thermal share by lagged provincial coal-consumption rate. Each outer fold independently tunes regularization with inner region-heldout validation.

The Ember mix-only baseline has region-equal MAE 0.05112 kgCO₂/kWh. Adding the generation coal-consumption rate lowers it to 0.04692 (−8.2%); supply coal-consumption rate lowers it to 0.04722 (−7.6%). Province-weighted MAE falls 0.04898→0.04519/0.04482. For the generation-rate model, MAE improves in both target years (2022: 0.05994→0.05480; 2023: 0.04229→0.03905) and region-equal P90 improves in both (0.09255→0.08475; 0.06691→0.05984). The paired seven-region cluster bootstrap for the pooled region-level MAE difference is [−0.00797, −0.00043], with 4/7 regions improved. This narrowly meets the frozen exploratory gate for Ember generation-rate adjustment.

For Ember supply-rate adjustment, MAE improves in both years and 5/7 regions, but its paired bootstrap interval [−0.00836, +0.00029] crosses zero. For NBS mix, both rate adjustments improve pooled MAE in both years and 4/7 regions, but the intervals cross zero (generation: [−0.01274, +0.00075]; supply: [−0.01123, +0.00179]). NBS’s generation-rate model also improves region-equal MAE 0.05450→0.04952; the evidence remains low-power and does not establish a stable population gain.

## Interpretation and limits

This is evidence that the annual local thermal-efficiency proxy can add information beyond a same-source, nested-regularized mix model, especially for the Ember comparison. It is a candidate data/structure contribution, not yet an algorithmic novelty claim: the transformation is simple, the cluster count is seven, the best interval barely excludes zero, and the NBS-only comparison is inconclusive. Both metrics and thresholds were frozen before the run.

CEC yearbook values are retrospective. Their release dates have not been verified against a real forecast origin; the 2024 yearbook cannot be treated as information available at the 2022/2023 target dates. The targets are annual MEE consumption-side factors, not hourly CIF; no interval coverage is measured. The heat rate applies to larger plants and is not itself a CO₂ emission factor. The result does not yet validate the user's eventual monthly all-generation-type schema, whose exact headers and vintage are still needed for a data-contract audit.

Next: first lock down actual category definitions, units, cumulative-vs-monthly convention, scope, and release dates for the user-provided monthly table; then test heat-rate information under a publication-time-valid forecast origin. If recent annual rates cannot be legally aligned to target dates, treat them as a retrospective explanatory proxy and focus the China benchmark contribution on a leakage-safe, multi-source information/identified-set evaluation rather than predictive accuracy claims.

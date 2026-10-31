# V94 protocol: high-threshold stress test for coarse thermal information

Frozen 2026-10-28. Extend the China-style monthly technology-margin coarsening experiment to thresholds 600 and 900 g/kWh, where V54's thermal EF support (400–1000) straddles the threshold. Use the same V90 358-region-month retrospective cohort, 25 previously opened balancing authorities, 2024–2025 periods, and ≥1/5/10% K targets.

## Frozen comparisons

1. Detailed EIA monthly fuel margins without capacity.
2. Coarse monthly thermal/nuclear/hydro/wind/solar margins without capacity, with thermal subfuel EF unknown within [400,1000].
3. The same coarse monthly fields plus lagged-year technology-group capacity, scored on all 358 frozen V90 months.
4. Detailed monthly fuel margins plus lagged-year fuel-specific capacity, reported only on its feasible intersection; never drop coarse-feasible months to force the fine-capacity comparison onto the smaller sample.

EIA hourly fuel records are used to create the coarse monthly margins as an oracle coarsening and to compute actual fixed-EF proxy K. The cohort is not expanded, and hourly labels do not train or select model parameters. All four candidate sets must contain observed K; the coarse monthly set must contain the detailed-margin set, coarse capacity set must nest within coarse no-capacity set, and coarse capacity must contain the detailed-fuel-capacity set. Any failure stops the run.

## Interpretation

Thresholds 600/900 make thermal fuel mix a potentially decisive hidden variable, unlike thresholds below all thermal EF support. If coarse capacity results lose efficiency, this identifies a concrete minimum-information gap: a local thermal carbon-factor or coal/gas composition summary may be needed for high-tail risk, whereas lower-threshold targets may need only coarse thermal generation. This remains a retrospective fixed-EF proxy experiment, not China validation or calibrated forecast coverage.

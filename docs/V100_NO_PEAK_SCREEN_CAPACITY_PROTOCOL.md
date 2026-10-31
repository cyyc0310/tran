# V100 protocol: lagged capacity without target-hour peak screening

Frozen 2026-10-28. V90 retained only months whose observed hourly group peaks were within lagged annual capacity. This is target-label-dependent cohort selection. V100 removes that filter to measure whether the coarse capacity set survives on the full candidate frame.

Use V90's exact 25-BA roster, 2024–2025 months, 2023/2024 lagged EIA-860 group capacity, eight monthly fuel margins, fixed proxy EF, thresholds 100/300, and tail rates 1/5/10%. Keep months with at least 80% hourly coverage and monthly group energy feasible under lagged group capacity; these filters use metadata and monthly totals only. Do not filter on observed hourly peaks or exceedance counts. Compare V66 monthly-margin sets with the same V90 relaxed capacity-group outer sets and their intersection.

For each held-in region-month and threshold, report whether the observed hourly K lies in each set. A capacity set that excludes K is a coverage failure, not a row to drop. Report interval widths, positive/negative/abstention decisions, and error rates for any positive/negative certificate; mark any decision whose set fails to contain K as invalid. Denominators must include the whole monthly-energy-feasible cohort. Also report the original V90 peak-screened cohort only as a labeled retrospective subset, not as the main estimate.

No unseen region is implied: this reuses already opened EIA BAs and the fixed-EF proxy. No result should be described as China validation or forecast calibration.

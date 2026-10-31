# V230 Frozen protocol: LP-sharp hourly CIF sets under EF uncertainty

Date: 2026-10-30. Status: frozen before computing the LP-sharp EF stress results.

## Purpose

V229 showed that a monthly-cap-only uncertainty envelope under a ±10% fixed-factor stress can be very wide. This experiment uses the same factor stress but solves the pointwise feasible-set extrema with linear programming, to separate genuine information loss from a loose triangle bound.

This is a sensitivity reanalysis on the already opened V226 EIA cohort, not a blind confirmation. The ±10% EF box is a stress scenario, not an empirical uncertainty interval.

## Data, target, and factor uncertainty

Reuse V226's 2026 July–September 26-BA/74-BA-month cohort, monthly margins, fixed-factor nominal CIF, source donors, and field selections. Hold hourly measurements exact. Compare EF half-widths `r_e=0` and `r_e=0.10`, with independent positive-category factors in `[e_j(1-r_e), e_j(1+r_e)]`; nominal zero factors stay zero.

For each hour, feasible fuel allocation `x` obeys monthly category margins, total generation, and the exact measured aggregate rows. Use the strongest valid per-hour category bounds `max(0,M_j-(M_total-G)) ≤ x_j ≤ min(M_j,G)`. Solve `min e_lowᵀx` and `max e_highᵀx` by linear programming. Full-eight observations fix every `x_j`; the factor interval then follows directly. These pointwise sets outer-bound any joint monthly allocation.

## Methods

1. `factor_span`: V228-selected minimum rows for the target month's active support.
2. `legacy_four`: total + thermal + coal + gas.
3. `full8`: exact eight hourly fuel categories.

For each method, the day score is the maximum endpoint distance of its hourly feasible CIF interval from the nominal monthly anchor. Calibrate with 12 random days (100 fixed-seed splits per BA-month); evaluate remaining days. Report nominal-proxy hourly/day coverage, robust-set containment (the prediction interval contains the whole hourly feasible CIF interval), width, and Winkler-90.

## Stop checks and limitations

- At `r_e=0`, factor-span and full8 hourly intervals must agree within `1e-8`; the legacy result must reproduce the saved V228/V226 score within numerical tolerance.
- At least 30 BA-months and 10 BAs must remain eligible.
- Do not interpret this stress as China factor uncertainty; zero-factor categories, imports, temporal EF variation, and network flows are excluded.
- LP bounds exploit exact group readings but only per-hour monthly-margin caps, so they are sharp for this stated hourly feasible polytope and may be wider than globally joint-month sharp bounds.

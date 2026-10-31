# V95 protocol: one monthly coal-generation field as a minimum local statistic

Frozen 2026-10-28. Test whether adding only monthly coal generation to the China-style thermal/nuclear/hydro/wind/solar monthly table is enough to recover high-threshold hourly tail identification. Use τ=600/900, ≥1/5/10% targets, the unchanged 358-region-month V90 cohort, and the same one-year-lagged coarse technology capacities. The 326-month fuel-capacity-feasible intersection is reported separately.

## Information budgets

1. Coarse monthly group totals alone, thermal fuel mix unknown in proxy EF range [400,1000].
2. Coarse monthly group totals plus one monthly coal-MWh field; the residual thermal amount is noncoal EF in [400,800].
3. Each of the preceding budgets plus coarse annual group capacity.
4. Detailed monthly fuel margins and, where the full fuel-capacity audit passes, fuel-specific annual capacity.

The hourly EIA records are aggregated to create the monthly coal field and labels; this is an oracle coarsening, not a Chinese public-data forecast. A field is useful only if the resulting identified set shrinks while containing observed K. Added information must nest within the prior set. The single-coal-field method uses a bounded 3D polytope over event thermal output, event coal output, and event clean output; vertices are enumerated exactly with rational arithmetic. Event thermal EF uses coal=1000 and noncoal upper EF=800; complement thermal EF uses coal=1000 and noncoal lower EF=400.

## Interpretation boundary

This is a minimum-statistic mechanism experiment. It does not establish that the monthly coal field is publicly available with the same provincial boundary, or that capacity is historically complete. No target-hour blind selection, actual CIF validation, or new-region forecast is included.

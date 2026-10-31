# V271 protocol: temporal information value under asymmetric fuel-factor support

Date: 2026-10-30. This is the first joint set-bound experiment using the V269 empirical EF endpoints; it is not China validation.

## Question

When fuel-specific EF levels are uncertain within the V269 asymmetric intervals, how much does exact daily total generation and exact hourly total generation still reduce the monthly hourly tail-count identified set?

## Frozen cohort and set

- Use the same 16 EIA BA-months from V240/V264, year 2025, threshold 500.0001 g/kWh.
- Use the quarter-specific V269 90%-generation intervals for coal, oil/petroleum coke, and gas; non-fossil direct EF remains zero.
- Treat each factor interval as a shared month-level parameter, not a separate hourly EF. The risk-count objective is coordinatewise nondecreasing in each fuel EF, so the robust lower endpoint is solved at the all-lower EF vector and the robust upper endpoint at the all-upper vector.
- Compare three nested information budgets on the same cases: monthly fuel margins only (analytic sharp bounds), exact daily generation totals, and exact hourly generation totals.
- Require all endpoints to prove optimality. Report incomplete cases separately. Verify set nesting, containment of the fixed-project-EF proxy tail count, and monotone width reduction across information budgets.

## Limits

V269 factors are U.S. state-quarter empirical supports transferred to already-open EIA BA compositions. The 90% generation support is not a target confidence set. Hourly CIF labels are fixed-factor EIA proxies. This measures identification-set geometry, not forecast coverage or Chinese operational performance.


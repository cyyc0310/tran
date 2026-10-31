# V231 Frozen protocol: what monthly fuel margins identify without local hourly mix

Date: 2026-10-30. Frozen before computing V231 results.

## Question

The user reports that public monthly tables include all generation categories. This experiment asks what that monthly information identifies about *hourly* CIF before any local hourly fuel-group reading is collected, and quantifies the value of the minimum V228 factor-span readings.

## Cohort and target

Reuse V226's already-opened EIA proxy cohort: 2026 July–September, non-development BA roster, 26 BAs and 74 BA-months. Monthly generation margins and nominal fixed EFs are provided. The EIA hourly total-generation denominator is temporarily treated as observed to make the no-group-information case optimistic; it is not implied by the user's monthly-only data.

## Information regimes

1. **Monthly margins + hourly total only (optimistic monthly-only proxy):** for each hour, each fuel amount is bounded by monthly availability and the known hourly total. Find minimum and maximum emissions by allocating feasible generation across fuels in EF order.
2. **Monthly margins + V228 minimum factor-span group readings + hourly total:** compute the exact nominal CIF identified by the span channels.
3. **Monthly margins + full-eight hourly fuel readings:** exact proxy label.

At fixed EFs, calculate the width of the per-hour feasible CIF set and summarize mean, median, 90th percentile, maximum, and whether the nominal proxy CIF lies inside. Do not interpret pointwise intervals as simultaneous full-month guarantees; monthly hour allocations are not jointly optimized.

## Interpretation gates

- At least 30 BA-months and 10 BAs must remain.
- Every nominal proxy CIF must be inside its monthly-margin feasible set within `1e-8`.
- Factor-span and full-eight widths must be below `1e-8` when their readings are supplied.
- Report that actual China month-only tables also lack the hourly total used in regime 1; regime 1 therefore understates, rather than overstates, the information gap.

## Limits

This is a fixed-EF EIA generation-derived proxy, not measured emissions or China validation. The LP bounds are sharp for each hour's marginal feasible set, but do not enforce joint allocation across all hours. No forecasting, calibration, or out-of-region performance claim is made here.

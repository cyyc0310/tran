# V225 Partial local fields with sharp monthly-margin risk bounds

Date: 2026-10-30. Status: development experiment on the same EIA proxy cohort as V215–V224.

## Problem

A sampled day may not expose all eight hourly fuel categories. Monthly category totals and a small number of hourly group totals nevertheless constrain the unknown hourly fuel allocation. The experiment asks whether these physical constraints can produce a useful upper bound on each sampled day's maximum hourly CIF deviation, allowing day-level risk calibration with fewer observed fields.

## Feasible hourly set

Let M_j be the target month's total generation of fuel j, G_h the observed hourly total generation, and x_{j,h} the unknown hourly generation. For each hour, the monthly margins imply

- lower bound `max(0, M_j - (M_total - G_h))`,
- upper bound `min(M_j, G_h)`.

Observed group fields add linear constraints, such as the coal amount or thermal sum. Under the frozen project EFs, the minimum and maximum hourly emissions are linear programs over these box constraints and fixed group sums. The implementation solves each group by allocating remaining generation to the lowest or highest EF first; this is exact for the separable linear objective. The fixed-factor CIF interval is the emission interval divided by observed total generation.

For sampled day d, define `U_d = max_h max(|L_h - c|, |U_h - c|)`, where `[L_h,U_h]` is the sharp feasible CIF range and c is the monthly fixed-factor anchor. This bounds the unknown day's true maximum hourly absolute deviation from c. The split-conformal radius is the finite-sample 90% order statistic of sampled day-level U_d values. The reported interval is `[c-q,c+q]` for all unsampled hours/days. Random day sampling without replacement gives a design-based marginal statement for a randomly selected held-out day; it is not simultaneous coverage of the entire month.

## Field sets and comparison

Target-month margins for all eight fuels are known post-month. At each hour of a sampled day, compare:

- `full_8`: exact eight hourly fuel values;
- `total_only`: hourly total generation;
- `total_plus_coal`: hourly total and coal;
- `total_plus_thermal`: hourly total and thermal total;
- `total_thermal_coal`: hourly total, thermal total, and coal;
- `total_thermal_coal_gas`: hourly total, thermal total, coal, and gas.

Use 7, 9, 12, and 15 random local days; 100 random partitions per region-month. All methods retain the same monthly anchor; no local sample changes the center. Compare to source-only leave-region-out daily-maximum calibration and exact eight-field local calibration. Paired intervals resample the 24 development regions.

## Scope

The local fields are hourly readings through an entire day, not one daily scalar. EIA labels use reported hourly fuel generation and fixed factors, not independently metered emissions. The experiment develops a CIF-specific feasible-score construction; it does not claim a new generic conformal algorithm.

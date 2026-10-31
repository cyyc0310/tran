# V223 Sparse daily fuel snapshot risk calibration protocol

Date: 2026-10-30. Status: value-of-information experiment on Carbon Monitor daily estimates; no independent metered-CIF validation.

## Question

Given exact target-month generation totals for all eight categories, can a small number of random target-day category-generation snapshots constrain the remaining daily CIF risk without changing the month-level center?

## Dataset and split

- Carbon Monitor Power global daily generation CSV, SHA-256 recorded in `results/v223-sparse-daily-risk-snapshots-r1/metadata.json`.
- Source calibration years: 2019–2023. Target years: 2024–2025; 75 held-out countries/aggregates and 1,800 complete country-months.
- For each target country, the source calibration pool excludes all history from that country.
- Fixed factors: coal 1000, gas 400, oil 800, other 400, and nuclear/hydro/wind/solar 0 gCO2/kWh. Daily labels are mechanically derived from the same source generation data.
- The center is frozen to the target month's generation-weighted fixed-factor mean, known only after that month closes.

## Information budgets and interval rule

Within each target month, sample 7, 8, 9, 12, 18, or 21 calendar days uniformly without replacement. Each sampled day supplies its full eight-category generation vector, allowing its daily fixed-factor CIF proxy to be calculated. Use absolute residuals from the unchanged monthly anchor to construct a symmetric 90% split-conformal interval; score only unsampled days. Repeat 100 random partitions per country-month-budget.

The finite-sample conformal radius is the `ceil((n+1)*0.9)` order statistic. It is infinite when this rank exceeds the calibration sample size; hence budgets 7 and 8 abstain with an unbounded interval, and 9 is the first tested finite radius. Random sampling without replacement supports a marginal random-day coverage interpretation over the random partition design. It does not imply simultaneous coverage of every remaining day, conditional coverage for every country, or operational guarantees for a fixed set of sampled days.

The comparator is a single source-only conformal radius computed from 2019–2023 daily absolute deviations from each source month’s own exact monthly anchor, with the target country removed. This source pool is large and its target coverage is empirical, not an exchangeability guarantee across countries.

## Metrics

For each partition, compute coverage, interval width, and Winkler-90 over unsampled days. Average partitions within country-month, months within country, then countries equally. Paired uncertainty intervals resample the 75 countries (20,000 draws).

## Limitation

Carbon Monitor daily generation is estimated secondary data. The [2023 China method paper](https://www.nature.com/articles/s41597-023-02094-2) describes daily disaggregation of monthly fuel generation with daily coal-consumption patterns from eight coastal provinces. Daily labels and local snapshots in this test therefore cannot serve as independent Chinese metered validation.

# V271 results: the EF gate predicts when hourly information has no robust value

Date: 2026-10-30. This is a partial-identification experiment on an already-open U.S. EIA cohort, not a forecast-coverage result or China validation.

## Design and proof status

The 16 V240 BA-months were evaluated at a 500 g/kWh threshold under V269's fuel-specific, quarter-specific asymmetric CEMS support intervals. Each EF is a shared month-level parameter. Since the tail count is coordinatewise nondecreasing in every fuel EF, the exact robust minimum uses all lower EF endpoints and the exact robust maximum uses all upper endpoints. This avoids treating each hour's factor as independently varying.

For each case, I compared monthly fuel margins only, exact daily total generation, and exact hourly total generation. All 64 endpoints that define the primary daily/hourly robust intervals proved optimal. Three of 64 extra monotonicity-diagnostic endpoints did not prove and are not used. All 32 proved daily/hourly sets contain the fixed-project-EF proxy tail count and nest within their monthly-only set.

## Matched information-budget frontier

| Information available | Mean robust tail-count width | Reduction from monthly-only |
|---|---:|---:|
| Monthly fuel margins | 734.1 h | — |
| Exact daily generation totals | 727.8 h | 6.4 h |
| Exact hourly generation totals | 589.3 h | 144.9 h |

On the paired daily-to-hourly comparison, exact hourly totals reduce mean width by **138.5 h (19.0%)**. Fourteen of 16 cases narrow. PACE July and October are the only two cases whose factor-support weighted monthly mean straddles 500 g/kWh; both retain the full `[0, 744]` count range, and exact hourly totals provide no reduction. This matches the proportional-allocation non-identification theorem. For the other 14 months the EF support stays on one side, and exact hourly totals narrow the set by 8–362 hours, though the resulting intervals remain wide.

All robust lower count endpoints are zero in this cohort: temporal information narrows upper-tail count uncertainty but does not certify a positive minimum tail count. The nominal proxy's observed count is contained in every proved set by construction; this is not predictive coverage.

## Why this is the strongest current candidate

This is the first joint experiment in the search that links factor uncertainty shape to the **measured marginal value of time information** under one matched set of targets. The result suggests a two-stage information rule: calibrate target fuel-factor support first; if the weighted factor set crosses the decision threshold, collecting more hourly totals cannot narrow the worst-case tail-count set; if it does not cross, measure temporal totals and quantify their remaining value. The statistical level and support shape matter, so one global ±r factor box can direct measurement effort to the wrong months.

This is a credible CIF-specific method hypothesis, not yet a breakthrough claim. EF supports are empirical U.S. CEMS state-quarter ranges transferred to EIA BA mixes, the hourly response is fixed-EF proxy data, and the monthly cohort has already been inspected. No probabilistic forecast or held-out regional coverage has been tested. The next hurdle is a preregistered outer-region study with target-justified fuel-factor supports and matched strong baselines; preserve unseen BA data until the EF contract is frozen.

## Reproducibility

- Protocol: `docs/V271_ASYMMETRIC_EF_TEMPORAL_VALUE_PROTOCOL.md`
- Runner: `scripts/v271_asymmetric_ef_temporal_value.py`
- Endpoint tables and metadata: corrected-chain rerun `results/v271-asymmetric-ef-temporal-value-r2/`. The r1 run used the same valid CEMS factor endpoint table, but r2 directly consumes the corrected V269 artifact.


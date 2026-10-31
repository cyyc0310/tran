# V131 — Daily Risk-Side Information Value

## Question

After a monthly threshold-side fuel statistic has been frozen, how much can a minimal daily local summary constrain the monthly number of hours above an emissions-intensity threshold?

## Frozen design

- Reuse the exact 358 region-month cohort from V129 and the same EIA hourly fuel records, factor intervals, strict thresholds (600 and 900 gCO₂/kWh), and 1%, 5%, and 10% monthly count decisions.
- Do not tune a model or choose regions/months using the outcomes. For each observed hour, calculate the nominal fixed-factor event label only as a containment audit.
- Compare four nested information contracts: (1) one monthly risk-side energy amount plus monthly thermal/clean totals; (2) daily risk-side amount plus daily thermal and clean totals; (3) daily detailed fuel totals; and (4) monthly detailed fuel totals. Capacity is excluded in every arm so the only difference is temporal/fuel information.
- Daily summaries use UTC calendar-day bins. Daily energy totals are derived by aggregating hourly EIA records; a bin has the number of valid observed hours in that source. All arms score the same observed hours.
- At τ=600, the risk-side field is coal+oil MWh; gas+other fossil is the remaining thermal amount. At τ=900, it is coal MWh, with the EF endpoint at 900 retaining K=0 in the robust union. The nonthermal groups have zero proxy EF under the stated stress assumptions.
- Compute each daily interval with the V129 exact threshold-side solver, then sum lower and upper endpoints across days. This is a conservative outer interval because the daywise factor-support hull does not exploit cross-day dependence in a common unknown emission factor.
- Assert that the monthly nominal observed K lies inside all monthly and daily risk-side intervals and that the daily risk-side interval contains the summed daily detailed robust interval. Failures are retained and invalidate the run.

## Interpretation boundary

This is retrospective oracle coarsening of EIA hourly fuel generation, not a Chinese public-data evaluation, a forecast, or an actual CIF label study. The EF intervals are sensitivity assumptions, and daily release availability is not established. The experiment estimates the value of a candidate three-number daily information contract over the monthly contract; it does not claim that these fields are currently public or the unique minimal contract.

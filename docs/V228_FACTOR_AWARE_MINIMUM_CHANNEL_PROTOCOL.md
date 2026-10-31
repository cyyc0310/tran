# V228 Frozen protocol: factor-aware minimum hourly aggregate channels

Date: 2026-10-30. Status: frozen before scoring the new field rule on V226's held-out EIA cohort.

## Hypothesis

The minimum hourly aggregate fields needed to identify generation-side fixed-factor CIF are determined by whether the factor vector lies in the span of the measured group fields, restricted to fuels with positive target-month margins. A fixed set of named fuel channels may therefore waste measurements or fail to identify CIF.

## Frozen selection rule

For a target month, let `S` be fuel categories with positive monthly generation. Let `e_S` be their fixed emission factors. Candidate hourly measurements are group sums for `thermal={coal,gas,oil,other}`, and individual `coal`, `gas`, `oil`, and `other`; total generation is always measured as the denominator. Enumerate subsets by increasing channel count, with stable tie-break order `thermal, coal, oil, gas, other`. Select the first subset whose measurement-row span contains `e_S` to absolute residual tolerance `1e-9`. If no subset identifies the numerator, stop and report the rule infeasible; do not fall back to a post-hoc choice.

This makes the target-month fuel margins part of the input contract: a category with exactly zero monthly margin is inactive and its channel need not be sampled. For the current EIA factors `[coal=1000, gas=400, oil=800, other=400, clean=0]`, expected typical channels are `thermal+coal+oil`; if oil is absent, `thermal+coal` suffices. The channel derivation is algebraic, not a learned model.

## Cohort and comparison

Use the V226 frozen outer cohort: 2026 July–September, BAs outside the V224/V225 development roster, with inclusion based only on completeness and source availability. The cohort and data have been opened for V226; V228 is a pre-specified method-specific reanalysis, not a new blind confirmation.

At 12 uniformly sampled target days per BA-month, 100 fixed-seed repetitions, calibrate whole-day maximum absolute error around the same monthly fixed-factor anchor. Compare:

1. `factor_span_minimum`: the rule-selected fields; its channel linear combination reconstructs each sampled hour's exact proxy emissions numerator and denominator.
2. `legacy_total_thermal_coal_gas`: V226 four-field feasible-bound score.
3. `full8_local`: exact eight-fuel hourly score.
4. `source_loo`: historical same-month source risk pool, excluding target BA.

All methods share random day splits for paired comparisons. Report observed field count, hourly coverage, simultaneous 24-hour day coverage, interval width, Winkler-90, and paired BA-bootstrap intervals.

## Gates

- At least 30 BA-months and 10 BAs remain eligible.
- The span rule must reproduce the exact hourly score within `1e-8 gCO2/kWh` for every selected hour; otherwise stop.
- It must use fewer hourly numeric fields than the fixed four-field method in at least 20% of BA-months.
- Its macro day coverage must be at least 90%, and its width/Winkler must equal the full-eight local method within numerical tolerance.
- No claim of blind confirmation, metered CIF, or Chinese operational validity.

## Scope and limitations

This is an information-design result under fixed factors and post-month margins. If the true target includes time-varying fuel efficiency, imported electricity, storage, losses, or a consumption-side network-flow factor, the group-row span may not identify the target. The hourly aggregates are not part of the user's confirmed public monthly-only information budget. Generic constrained regression, temporal disaggregation, and conformal calibration are not claimed as novel.

Frozen target data source: native EIA 930 2025–2026 files recorded in the V226 protocol. Source and protocol hashes are saved in the V228 output metadata.

# V371 protocol: composition-weighted donor shape transfer

## Question

Can the local five-class monthly mix (thermal, hydro, nuclear, wind, solar) improve donor selection for hourly CIF shape transfer compared with equally averaging every donor region?

## Data and target

- Use the seven-region 2022–2024 EIA proxy panel and outer leave-one-region-out folds.
- Aggregate local fuel margins to the five classes published by the NBS page; retain the unreported EIA `other` as an explicit residual class so feature shares sum to one.
- The held-out region's only local information is its same-month composition vector and monthly thermal share. Its hourly CIF proxy is never used for donor selection.
- The target is the fixed-factor generation CIF proxy, not measured emissions.
- Use thermal EF=700 g/kWh for the local level and zero nonthermal direct intensity; this keeps level identical across shape methods.

## Frozen donor-shape methods

For each donor region, center its hourly CIF proxy by its own month mean. Predict the held-out target's hourly shape deviation using:

1. `equal_all`: equal weight over all available donor regions;
2. `nearest3`: equal weight over the three donors with smallest Euclidean distance in the six-part monthly composition vector;
3. `kernel_015`: weights `exp(-d²/(2*0.15²))`;
4. `kernel_030`: weights `exp(-d²/(2*0.30²))`.

All methods add the same local thermal-share anchor `700*s_local`; `local_flat` is a diagnostic without hourly shape. Missing donor timestamps use zero shape deviation. No bandwidth or neighbor count is tuned on held-out outcomes.

## Evaluation

- Score hourly CIF proxy MAE and absolute hourly tail-count error at strict thresholds `τ ∈ {300,500,700}` g/kWh.
- Paired region-month comparisons; average within region first, then equally over seven regions. Report seven-cluster bootstrap percentile intervals (10,000 resamples, seed 371), region wins, and per-region scores.
- Sensitivity to EF is not part of this shape-isolation experiment; V368 already shows EF scenario dependence.
- No coverage or interval claim.

## Stopping rule

Composition weighting is a candidate method increment only if it improves the local-level equal-all donor-shape baseline on tail-count error around τ=300, has a positive seven-region cluster interval or broad region wins, and does not cause severe degradation at τ=500. A null or unstable result closes this weighting route for the current data contract.

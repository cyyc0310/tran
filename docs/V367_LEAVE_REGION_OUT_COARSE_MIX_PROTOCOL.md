# V367 protocol: leave-region-out value of local monthly mix

## Question

Does local monthly generation composition improve hourly CIF reconstruction in an unseen region, beyond an otherwise identical donor-transferred hourly shape? This is an information-value experiment for the provincial-data direction from V366.

## Data and target

- Frozen input: the seven-region 2022–2024 EIA-derived hourly files in `results/research/data/`.
- Target: hourly positive reported-generation eight-bucket CIF proxy using the existing fixed factors `[1000,400,800,400,0,0,0,0]` g/kWh for coal, gas, oil, nuclear, hydro, wind, solar, and other.
- Exclude rows marked invalid, nonfinite, or with nonpositive total generation. Do not impute or relabel missing data.
- The target is a generated proxy, not direct emissions, not full-grid emissions truth, and not China validation.

## Information contract and baselines

For each target region and calendar month, use all other regions in the same year-month as the donor pool. The target region contributes only its month-level generation margins; its hourly outcomes are never used to form donor curves.

Compare two hourly predictors with the **same** leave-one-region-out donor hourly-shape deviation:

1. `pooled_level_plus_donor_shape`: donor pooled monthly CIF level plus donor hour-of-month deviation from that level.
2. `local_level_plus_donor_shape`: target region monthly CIF level, computed from its month-level fuel margins, plus that same donor hour-of-month deviation.

Also report pooled/local flat monthly-anchor predictors as diagnostics. The only difference in the paired principal comparison is whether the monthly CIF level comes from donor regions or the local monthly fuel mix.

## Evaluation

- Leave one region out; no hourly target values from the held-out region enter any donor statistic.
- Evaluation threshold events are strict `CIF > τ` at `τ ∈ {300,500,700}` g/kWh.
- Primary metrics: hourly CIF MAE and absolute error in monthly exceedance count, each averaged equally over region-months; also give region-level paired differences and a region-cluster bootstrap interval (7 regions, 10,000 resamples, seed 367).
- Report sample count, valid-hour count, valid fraction, and per-region results. Missing donor timestamp profiles fall back to the donor monthly level (zero shape deviation).
- No tuning on held-out outcomes, no parameter selection, and no claim of conformal or probabilistic coverage.

## Interpretation gate

Evidence for useful spatial information requires local-level improvement over pooled-level under the identical donor shape in paired region-month scores, with the sign broadly consistent across held-out regions and thresholds. Any improvement is a proxy-domain result only. A positive result does not establish China transfer, direct emissions accuracy, interval coverage, or superiority to operational systems; a null or mixed result weakens the claim that provincial monthly mix alone is a sufficient CIF innovation.

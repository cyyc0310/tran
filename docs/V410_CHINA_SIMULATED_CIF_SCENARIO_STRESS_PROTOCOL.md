# V410 protocol: release-lagged coarse mix under China hourly scenario stress

Frozen 2026-10-31 before parsing/scoring hourly outcomes from scenarios S2–S5. Scenario S1 was opened for feasibility and is development-only. The test labels are model-generated projections, not measured hourly CIF.

## Question and estimand

Can a source-trained, cross-region hour-of-year model that receives only a legally lagged public NBS five-technology mix improve projected hourly provincial carbon-factor prediction and threshold-count intervals over a source-only shape baseline, when the held-out target scenario changes load growth or carbon budget?

The output target is the Figshare study's *projected provincial hourly carbon emission factor* (published in tCO₂/MWh). Its article says these values come from power-system planning/operation simulation plus carbon-flow tracing. Convert values to gCO₂/kWh by multiplying by 1000 before scoring and applying thresholds. Treat this as a simulated consumption/network-flow reference, not the project's fixed-EF generation proxy and not actual hourly emissions.

## Frozen data and cohort

- Source: Figshare item `10.6084/m9.figshare.28953545`, version 3, CC BY 4.0. Use 2025 hourly sheets from S1–S5 and the versioned Figshare metadata/annotation. S1 may be used only to develop/freeze code and summary checks; S2–S5 are the scenario stress targets.
- Coarse features: the V394 National Bureau of Statistics current-value snapshot, using five generation categories (thermal, hydro, nuclear, wind, solar) divided by the same-month total. At target month `t`, use the `t-2` feature month to reflect the NBS release lag. Use only target months with all five `t-2` values and total present; for 2025 this is expected to be May–December, but the runner must derive and record availability rather than fill missing values.
- Include every mainland province with a complete hourly target series and legal coarse features. Do not remove provinces or months based on target CIF, residuals, threshold counts, or model scores.
- The current NBS historical API snapshot is not an as-published vintage archive. Therefore this is release-lag emulation, not fully certified real-time replay.

## Frozen prediction methods

For each target province `p`, fit on S1 labels from all other provinces only. Score that province on each of S2, S3, S4, and S5 over the common eligible target months.

1. `source_hour_profile`: mean S1 hourly factor across the 30 source provinces at each calendar hour; it receives no target-local mix.
2. `local_thermal_anchor`: constant hourly value `700 ×` the target's lagged NBS thermal share. The factor 700 is a fixed sensitivity baseline inside the previously used broad thermal support, not tuned on S2–S5.
3. `coarse_mix_ridge`: Ridge regression with fixed alpha=10, trained on S1 source hourly labels. Inputs are the five lagged NBS technology shares, sine/cosine hour-of-day, sine/cosine day-of-year, and sine/cosine annual phase. Do not tune alpha or add features after inspecting S2–S5 outcomes.

## Frozen interval and tail-count evaluation

For each outer target province, construct calibration residuals from S1 source provinces using leave-one-source-province-out predictions; no calibration prediction may use that source province's own labels. For each target calendar month, use the 90% finite-sample conformal order statistic of the source-region maximum absolute hourly residual in that month (rank `ceil(0.90 × (n+1))`, clipped to `[1,n]`). Form a constant-width simultaneous-within-month interval around each method's point prediction. Apply the identical calibration procedure to all three methods.

Report region-equal point MAE/RMSE, mean hourly interval width, region-month simultaneous coverage (all hours in the month covered), and worst-province coverage for each target scenario. At thresholds 300/500/700 gCO₂/kWh, derive the induced K-count interval `[count(L_h>τ), count(U_h>τ)]`; report its width, containment of the projected K, and positive/negative/abstention rates at 1/5/10% monthly tail targets. A failed simultaneous interval counts as a coverage failure and is never dropped.

## Interpretation gate

This is a **simulation-grounded spatial/scenario stress test**, not actual-China model validation. Do not pool scenarios as independent systems or bootstrap provinces as if numerous independent grids existed. A useful result requires the coarse-mix method to improve interval efficiency over `source_hour_profile` while maintaining monthwise coverage across the scenario targets, and the gains must not come only from one region or one scenario. `local_thermal_anchor` is a simple physical-level comparator. Even if this gate passes, the output cannot establish coverage for real China or resolve the NBS/CEF estimand and historical-vintage mismatch.

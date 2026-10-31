# V269 results: asymmetric empirical EF support changes threshold-gate decisions

Date: 2026-10-30. Cross-system stress test using 2025 U.S. CEMS factors and the already-open V240 EIA BA-month cohort.

## Fuel-group support intervals

For each fuel and quarter, I computed the narrowest interval containing at least 90% of state-quarter gross generation. The intervals are asymmetric and retain the observed factor level:

| Group | Quarter | 90%-generation support (g/kWh) | States |
|---|---:|---:|---:|
| Coal | Q1 | 863.6–1033.2 | 37 |
| Coal | Q2 | 873.2–1029.1 | 36 |
| Coal | Q3 | 858.2–1016.9 | 37 |
| Coal | Q4 | 875.6–1013.6 | 36 |
| Gas | Q1 | 377.4–434.1 | 47 |
| Gas | Q2 | 388.7–440.8 | 47 |
| Gas | Q3 | 392.6–455.2 | 47 |
| Gas | Q4 | 373.5–441.4 | 47 |
| Oil / petroleum coke | Q1 | 636.8–855.2 | 10 |
| Oil / petroleum coke | Q2 | 619.9–870.6 | 10 |
| Oil / petroleum coke | Q3 | 647.1–868.8 | 10 |
| Oil / petroleum coke | Q4 | 633.5–828.1 | 10 |

The intervals are generation-weighted state-quarter supports, not uncertainty intervals for a specific target region. Oil has only 10 states and much lower generation than coal or gas.

## Target-mixture threshold gate

Applied to the same 16 EIA BA-month fuel mixes at 500 g/kWh, with **all-source generation in the denominator**, the weighted-shortest asymmetric intervals crossed the monthly mean threshold in **2/16** cases (PACE July and October). V268's symmetric CEMS-centered 90%-generation ranges also crossed **2/16** (PACE July and October); the project-centered uniform ±10% gate crossed a different set of **4/16** (AZPS October and PACE April/July/October).

Thus uncertainty-set shape and center change which BA-months are judged threshold-ambiguous. The gate cannot be summarized by one universal relative tolerance. This is an empirical design signal for a candidate method, not evidence that the asymmetric interval is calibrated for a Chinese target.

## Correction to initial run

Initial r1 gate results accidentally divided by fossil generation alone. The empirical factor interval endpoints themselves are unaffected, but the monthly mean intensities and threshold-crossing flags were wrong. The corrected gate analysis is in r2 and supersedes the initial `ba_month_gate_comparison.csv`.

## Reproducibility and limits

- Protocol: `docs/V269_ASYMMETRIC_EMPIRICAL_EF_SUPPORT_PROTOCOL.md`
- Runner: `scripts/v269_asymmetric_empirical_ef_support.py`
- Full fuel intervals, corrected BA-month comparison, and metadata: `results/v269-asymmetric-empirical-ef-support-r2/`. The r1 run is retained as superseded.

All source-factor records are U.S. CEMS gross generation; target compositions are opened EIA balancing areas. No hourly CIF prediction was issued, no coverage was assessed, and no unopened target was used.


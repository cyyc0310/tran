# V267 results: measured CEMS factor dispersion is fuel-specific

Date: 2026-10-30. Cross-system stress test only; not a China factor estimate.

## Empirical CEMS intervals

From 10,441,281 measured 2025 CEMS unit-hour rows, I aggregated state-quarter factors for coal, natural gas, and oil/petroleum coke. The per-fuel, per-quarter radius is the smallest symmetric percentage around that quarter's national generation-weighted factor containing at least 90% of observed state-quarter gross generation.

| Group | Q1 radius | Q2 | Q3 | Q4 | Approximate center (g/kWh) |
|---|---:|---:|---:|---:|---:|
| Coal | 13.0% | 11.6% | 10.0% | 9.8% | 914–924 |
| Gas | 7.5% | 6.8% | 8.5% | 9.2% | 408–419 |
| Oil / petroleum coke | 14.8% | 27.6% | 24.8% | 21.5% | 681–748 |

The oil group has only 10 contributing states and much lower generation than coal or gas, so its radius is particularly unstable. These are dispersion envelopes over state-quarter means, not confidence intervals for a target region.

## Threshold-gate transfer stress

Applying those **group-specific relative radii around the project's nominal factors** to V240's same 16 EIA BA-month fuel mixes made 4/16 monthly mean intervals cross 500 g/kWh: AZPS October and PACE April, July, and October. V265's uniform ±10% box crossed the same 4/16. The monthly CIF denominator includes all generation, including non-fossil generation with zero direct EF.

## Correction to initial run

The initial r1 gate calculation omitted zero-direct-EF generation from the target CIF denominator and is superseded. The CEMS fuel-factor summaries and 90%-generation radii are unchanged; only target gate calculations were affected. The authoritative corrected run is r2.

Reproducibility: `scripts/v267_cems_calibrated_threshold_gate.py`; corrected results and inputs in `results/v267-cems-calibrated-threshold-gate-r2/`. Initial r1 is retained as a superseded run, not as evidence.

## Limits

This uses measured U.S. fossil CEMS gross load and transfers state-quarter dispersion to already-open EIA balancing-area mixes. It is not Chinese EF uncertainty, does not identify the target-region factor, and does not validate hourly predictions. V268 separately tests the factor-level shift.


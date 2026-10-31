# V98: measurement-error sensitivity of the threshold-side monthly statistic

## Question and frozen comparison

Does the V97 near-sufficiency result survive errors in the risk-side monthly MWh field? We expanded the observed side total to a symmetric relative uncertainty interval of ±1%, 2%, 5%, and 10%, clipped to the monthly thermal total, and took the union over all feasible latent values. The cohort remains the same 358 EIA balancing-area months used by V90/V97; annual coarse technology capacity is held fixed. The comparator is the detailed monthly fuel-margin set with exactly the same capacity constraints and rows. Thresholds are 600 and 900 gCO2/kWh, with 1%, 5%, and 10% monthly-hour tail targets.

## Results

| Threshold | Side-stat error | Mean side-set width (h) | Detailed-margin width (h) | Side positive certificate, 1% target | Side negative certificate, 1% target | Abstention |
|---:|---:|---:|---:|---:|---:|---:|
| 600 | 0% | 214.19 | 211.17 | 4.29% | 56.01% | 39.71% |
| 600 | 1% | 214.65 | 211.17 | 4.29% | 56.01% | 39.71% |
| 600 | 2% | 215.09 | 211.17 | 4.05% | 56.01% | 39.95% |
| 600 | 5% | 216.43 | 211.17 | 3.33% | 56.01% | 40.66% |
| 600 | 10% | 218.53 | 211.17 | 2.62% | 56.01% | 41.37% |
| 900 | 0% | 142.17 | 137.25 | 0.00% | 70.35% | 29.65% |
| 900 | 1% | 142.44 | 137.25 | 0.00% | 70.35% | 29.65% |
| 900 | 2% | 142.67 | 137.25 | 0.00% | 70.35% | 29.65% |
| 900 | 5% | 143.38 | 137.25 | 0.00% | 70.35% | 29.65% |
| 900 | 10% | 144.53 | 137.25 | 0.00% | 70.35% | 29.65% |

At ±10%, average width grows by 4.34 h at threshold 600 and 2.36 h at threshold 900 relative to the exact side field. The corresponding 1% positive-certification rate at threshold 600 falls by 1.67 percentage points; at threshold 900 it is zero even with exact side totals. The side statistic remains slightly less informative than all detailed monthly fuel margins at every error level. Thus V98 supports moderate sensitivity to side-total reporting error, not a forecasting gain or a new empirical breakthrough.

## Validity checks and limits

For every scored row, the computed actual hourly exceedance count lies in the set; each uncertain set contains its zero-error set and is nested in the coarse thermal-total-plus-capacity set. The zero-error solver exactly reproduces V96. The interval models uncertainty only in the side MWh field while treating broad thermal generation and coarse capacity as exact. It is a sensitivity envelope, not an empirically estimated Chinese reporting-error distribution. The EIA-derived, retrospectively capacity-screened cohort, fixed emission-factor proxy, and lack of an independent Chinese hourly CIF label remain decisive limitations.

## Reproducibility

- Protocol: `docs/V98_SIDE_ENERGY_MEASUREMENT_ERROR_PROTOCOL.md`
- Script: `scripts/v98_side_energy_measurement_error.py`
- Frozen output: `results/v98-side-energy-measurement-error-r2/`
- Protocol SHA-256: `0df14f544e1e06c21c9f302d56c66641c7a274972b1e4799b317136f63fd770f`
- Script SHA-256: `953698c70fa3a2c6c657e0a3a3c0c76bcdea6452e6176f9070ec56a5ca3239bf`
- Summary SHA-256: `fd606cb8801ca2ffcee0a72abb6ed0e8a34e9c4f9c9f2acd321c8504b5d71407`

# V113 results: future-time confirmation on 2026Q3

## Cohort and integrity

The EIA-930 2026 Jul–Dec file was downloaded from the official six-month-file endpoint after the protocol freeze. The file reports `Last-Modified: 2026-10-27 14:42:55 GMT`, SHA-256 `48005f09be801598e28f38079056260a0d07ab9a6d1a0324e8e92ca470b3c281`, and size 31,105,441 bytes. The target window was truncated at 2026-10-01, so only July–September were scored. CPLE/FPC/IPCO/PNM/SOCO each had complete coverage in all three months; all 15 slots also passed lagged group-energy feasibility.

The frozen V112 worst-source-region anchor, V103 detailed-fuel ±10% EF identified set, and the same set plus 2025 shared group capacity were applied without target-season recalibration. All three methods' intervals contain the low-to-high factor-box proxy K endpoints in 15/15 region-months at both thresholds.

## Future-time frontier

| Threshold | Robust anchor width | Monthly-only robust set width | Capacity robust set width | Capacity certificates |
|---|---:|---:|---:|---|
| 600 gCO₂/kWh | 341.3 h | 735.0 h | 447.3 h | All abstain |
| 900 gCO₂/kWh | 262.0 h | 588.0 h | 277.5 h | 20% negative at 1/5/10% targets; all others abstain |

No method issues a positive certificate. At τ=600, the actual nominal proxy K is zero in most rows, but factor-box upper K reaches 80 hours in PNM September and 12 in FPC September; the robust intervals appropriately retain abstention. At τ=900, the factor-box K range is zero for every row. The capacity method's 20% negative certificates all come from IPCO, where zero K is ruled out by the high-threshold support/capacity constraints; the empirical anchor remains abstaining because its worst-region calibration radius is broad.

## Interpretation

This is the first genuinely future-time confirmation in this sequence. It reproduces the H1 coverage-efficiency pattern: the robust anchor is narrower, while adding physical capacity provides a few selective negative certificates. The result is still weak evidence for a forecasting breakthrough because it covers only one quarter, five already explored regions, and a high-threshold target with zero observed proxy events. The τ=600 months include some factor-box tail risk but neither approach can safely classify them. No Chinese hourly CIF or month-ahead performance is validated.

## Reproducibility

- Frozen protocol: `docs/V113_2026Q3_FUTURE_TIME_CONFIRMATION_PROTOCOL.md`
- Runner: `scripts/v113_2026q3_future_time_confirmation.py`
- Outputs: `results/v113-2026q3-future-time-confirmation-r1/`
- Official source: [EIA-930 2026 Jul–Dec balance file](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2026_Jul_Dec.csv); official format/scope described in [EIA-930 documentation](https://www.eia.gov/electricity/930-content/about.html).
- Hashes, vintage and source audits are in `summary.json`.

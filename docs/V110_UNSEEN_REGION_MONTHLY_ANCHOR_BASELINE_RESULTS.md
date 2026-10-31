# V110 results: monthly-mean anchor versus identified sets

## Baseline and cohort

The nominal-factor point anchor uses only each target month's fuel margins: it assigns the monthly generation-weighted mean intensity to every valid hour. A region-balanced 90th empirical absolute-error radius is calibrated on 589 source-region-months from 25 development regions over 2024–2025, then evaluated once on the same 30 heldout region-months used in V104–V106. The calibration interval is descriptive; temporal and cross-region dependence prevent a formal conformal guarantee.

## Results

| Threshold | Anchor point MAE | Anchor interval width | Anchor interval coverage | Monthly-only robust set width | Capacity robust set width |
|---|---:|---:|---:|---:|---:|
| 600 gCO₂/kWh | 2.93 h | 45.8 h | 100% | 722.2 h | 434.1 h |
| 900 gCO₂/kWh | 0 h | 0 h | 100% | 577.6 h | 270.0 h |

At τ=600, the nominal anchor interval gives a negative certificate for the 10% tail target in all 30 months, while both robust partial-identification sets abstain on all 30. At τ=900, the nominal anchor gives negative certificates for all 30 months and all three event-rate cutoffs; the observed proxy K is zero in all 30. The capacity robust set certifies only 20.0% negative at 1%/5% and 23.3% at 10%.

## What this says—and does not say

On these five regions and the nominal fixed-factor proxy, the simple monthly-mean anchor is substantially sharper and more decisive than the worst-case identified sets. This is a serious challenge to any claim that the current robust interval method outperforms simple anchoring on observed proxy labels. The result may reflect a benign holdout mix, regime-specific near-zero tail rates, and nominal factors; 30 months do not establish future reliability. More importantly, V110 does not propagate EF uncertainty into the point anchor or calibration radius, while the set method uses a ±10% factor box. V111 is therefore required before making a fair robustness comparison.

The immediate paper implication is clear: a valid partial-identification benchmark must be compared with an empirically calibrated anchor at the same label/factor assumptions, and the robust-factor version of that anchor must also be evaluated. If the simple anchor remains accurate and narrow under the same EF uncertainty and across future periods, it defeats the practical value claim of the broad set method even if the set is mathematically correct.

## Reproducibility

- Frozen protocol: `docs/V110_UNSEEN_REGION_MONTHLY_ANCHOR_BASELINE_PROTOCOL.md`
- Runner: `scripts/v110_unseen_region_monthly_anchor_baseline.py`
- Outputs: `results/v110-unseen-region-monthly-anchor-baseline-r1/`
- Calibration rows, target rows, hashes and source audits are retained in the output directory.

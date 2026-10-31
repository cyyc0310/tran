# V368 results: local thermal share under the coarse China information budget

Protocol: [V368](V368_THERMAL_SHARE_ONLY_TRANSFER_PROTOCOL.md). Runner: `scripts/v368_thermal_share_only_transfer.py`. Results: `results/v368-thermal-share-only-transfer-r1/`.

## Setup

This ablation limits the held-out region's local monthly input to `coal+gas+oil / total generation`, matching the thermal-versus-other split available from the Chinese provincial monthly table. All other components of the two main predictors are identical: each receives the same hourly CIF shape deviation transferred from the other six regions. The paired difference is only the month-level thermal share, local versus leave-one-region-out pooled. The hourly test target remains the eight-bucket EIA fixed-factor proxy.

## Main result and sensitivity

There is useful information in the local thermal share around the lower `300 g/kWh` risk threshold, but the gain depends on the assumed common thermal effective factor:

| Thermal EF scenario | Tail-count MAE gain from local share at τ=300 | 95% region-cluster interval | Regions improved | Hourly CIF MAE gain |
|---:|---:|---:|---:|---:|
| 500 g/kWh | 188.6 h/month | 49.5–339.4 | 6/7 | 57.4 g/kWh (interval 8.6–126.9) |
| 700 g/kWh | 153.7 h/month | 16.2–356.3 | 7/7 | 69.4 g/kWh (interval −0.9–173.1) |
| 900 g/kWh | 119.6 h/month | 0.0–326.1 | 2/7 | 66.8 g/kWh (interval −38.5–215.0) |

Positive gain means the local-share predictor has smaller error than the pooled-share predictor, under the same donor hourly shape. At the primary midpoint EF scenario of 700 g/kWh, mean tail-count error fell from **541.2 to 383.4 hours**. At τ=500 and EF=700, local share was slightly worse on tail-count error (gain −12.3 h; interval −41.4–7.6; 4/7 regions improved). At τ=700, both methods predicted zero proxy tail hours, so there is no useful comparison.

The donor shape contributed little on its own: at EF=700 and τ=300, adding it changed local-anchor tail-count MAE from 395.8 to 383.4 h, while pooled-anchor error changed from 540.3 to 541.2 h. The main incremental signal came from the local monthly level. This is consistent with a level-versus-shape separation, but does not show a new forecasting algorithm.

## Scientific interpretation

This is a stronger, more relevant result than V367 because it respects a coarse local thermal-share input. It provides a cross-region proxy-domain indication that local monthly composition can improve a transferred hourly profile. The gain is threshold-sensitive and not stable to the thermal EF scenario: the 900 g/kWh scenario has only two region wins and bootstrap intervals including zero. This directly supports making the method **threshold-conditioned on local EF support**, rather than using one national factor or claiming universal gains.

The V368 evidence does not meet a breakthrough gate yet. It is based on seven already-opened U.S. balancing authorities, a generated CIF proxy, retrospectively reconstructed month totals, and one transferred-shape baseline. It has no Chinese hourly target, no formal coverage guarantee, no calibrated Chinese EF support, and no comparison of prediction intervals against conformal or conditional-quantile baselines. It is therefore an empirical motivation for the next interval experiment, not evidence of China transfer or deployable coverage.

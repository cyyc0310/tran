# V367 results: leave-region-out value of local monthly mix

Protocol: [V367](V367_LEAVE_REGION_OUT_COARSE_MIX_PROTOCOL.md). Runner: `scripts/v367_leave_region_out_coarse_mix.py`. Results: `results/v367-spatial-coarse-mix-proxy-r1/`.

## Result

Across 245 valid region-months, 175,560 hours, and seven EIA balancing authorities (2022–2024), the local monthly eight-fuel anchor plus a donor-transferred hourly CIF shape was compared with the same donor shape plus a leave-one-region-out pooled monthly anchor. The regional method reduced equal-region mean hourly CIF-proxy MAE by **110.0 g/kWh** (7/7 regions favored it; region-cluster bootstrap 95% interval 53.3–178.8).

At strict `CIF>300`, local detailed monthly margins reduced mean absolute monthly tail-count error by **460.5 hours** per region-month versus the pooled anchor (7/7 regions; 95% interval 290.8–586.3). At `CIF>500`, the gain was only **9.2 hours** (1/7 regions; interval 0–27.6). At `CIF>700`, neither method predicted any events and the score was uninformative.

This is a favorable result for spatially local monthly information, but it uses all eight local fuel margins. China's NBS table supplies a coarser thermal aggregate. V367 therefore motivated the stricter V368 ablation and must not be presented as a direct test of the China field budget.

## Limits

The held-out region's hourly proxy labels were excluded from its donor shape, but all seven regions had been opened previously; this is cross-validation, not an untouched confirmation. The hourly labels are fixed-factor generation proxies rather than direct emissions. Month margins are reconstructed retrospectively from the same generation files, without known publication vintages. No interval coverage claim follows from point-score gains.

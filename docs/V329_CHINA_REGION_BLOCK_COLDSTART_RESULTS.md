# V329 results: whole-region cold-start transfer of annual MEE factors

Run date: 2026-10-30. Frozen protocol: [V329 protocol](V329_CHINA_REGION_BLOCK_COLDSTART_PROTOCOL.md). Machine-readable outputs: `results/v329-china-region-block-coldstart-r1/`.

## Main result

The seven-region holdout contains 30 provinces and two forward target years. Both lagged composition models beat the outside-region training mean on pooled province-weighted MAE: 0.0746 kgCO₂/kWh for NBS and 0.0695 for Ember, versus 0.1227 for the mean. At equal-region weighting, MAE is 0.0757, 0.0715, and 0.1205 respectively. This supports a coarse annual composition signal under geographic cold start.

NBS did not reliably outperform Ember. Ember has lower pooled MAE by 0.00510 kgCO₂/kWh; the region-cluster bootstrap interval for NBS minus Ember is [−0.00260, +0.01466], with only two of seven regions favoring NBS. NBS beats Ember in North and Northeast; it is similar in Central, East, South, and Southwest, and substantially worse in Northwest. Thus replacing a third-party mix estimate with NBS-reported mix is not yet a robust transfer gain.

## Where transfer fails

- Northwest NBS MAE is 0.1953 in 2022 and 0.1608 in 2023, compared with Ember at 0.1463/0.1423 and the outside mean at 0.1477/0.1385. In the five-province fold, all NBS predictions have positive mean error (0.1953 and 0.1608).
- Northeast is a different failure: its outside-region mean is unusually strong. In 2023, NBS MAE is 0.1157 against 0.0354 for the mean; all three NBS predictions are worse than the mean. Both mix models overpredict this region.
- Southwest is an important counterexample to a blanket claim that regions with high renewable shares fail: both composition models have MAE near 0.03 while the outside mean is above 0.21.

The feature and prediction audit shows fitted coefficients can be unstable: for example, the 2021-trained Northeast wind coefficient exceeds 1.14 kgCO₂/kWh in both NBS and Ember NNLS fits. This signals weak factor identifiability under collinear, sparse group shares; it is not evidence that wind itself has that factor. A plausible but unproven explanation for the Northwest and Northeast bias is that local generation mix does not identify the consumption-side MEE target when interprovincial transfers and accounting boundaries differ. The present experiment cannot establish that cause.

## Interpretation and limits

This is useful evidence that lagged generation composition carries annual cross-region information, but it does not establish a universal cold-start model or an hourly CIF forecast. Targets are annual MEE consumption-accounting factors; only 2022 and 2023 are evaluated; there are seven correlated region clusters; feature vintages and release-time legality have not been verified; and NBS monthly production covers above-designated-size enterprises. The cluster interval is low-power.

The next analysis should test whether failures are predictable from training-only feature-support diagnostics and whether annual consumption/generation balance contributes information beyond composition. Any adjustment must be evaluated in an outer whole-region holdout, and should be treated as a candidate only if it improves both error and uncertainty calibration without using the held region's labels.


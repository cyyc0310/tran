# V50 coarse-mix conditional interval source-held-out test

Date: 2026-10-27. This evaluates whether the promising monthly mix → risk association translates into better intervals when an entire source region is held out. It uses only the ten V50 external calibration regions and does not test a new region.

## Design

The protocol was preregistered in `V50_COARSE_MIX_CONDITIONAL_INTERVAL_PROTOCOL.md`. For each of seven frozen outer point/risk functions, one of the ten external regions was fully held out. The other nine regions supplied both their hourly calibration scores and source-local p90 training responses. The test source response was used only to score final intervals.

Compared: (1) equal-region global 90th percentile on nine sources using `Scale-anchor + MCStd`, B0; (2) a source-level, one-feature linear p90 risk predictor using the hash-verified monthly full-mix feature; (3) global equal-region calibration of standard `HETScale`; and (4) the fixed raw anchor factor 2. No point/risk model was refit. Exactly 70 external prediction archives and ten external label packages were opened; no target/unseen source was read.

## Results

Metrics are averaged equally over the ten held-out source regions, with the seven outer functions first summarized within each source.

| Held-source method | Coverage | Width | Interval score | Lower miss | Upper miss |
|---|---:|---:|---:|---:|---:|
| Global equal-region `MCStd` | 86.86% | 623.2 | 833.7 | 0.00% | 13.14% |
| Monthly-mix conditional `MCStd` | 77.13% | 493.6 | 892.1 | 2.93% | 19.94% |
| Global equal-region `HETScale` | 87.30% | 641.6 | 884.0 | 0.07% | 12.63% |
| Raw anchor, fixed factor 2 | **88.50%** | 528.5 | **712.7** | 0.01% | **11.50%** |

The monthly conditional rule shrank intervals but sharply reduced coverage and had a worse interval score than all three comparators. It particularly failed on NWMT and WACM, with coverage of 29.7% and 28.6%; SCEG and SCL were also under-covered. Its macro coverage was 9.7 percentage points below the global `MCStd` q. The 90% target was not maintained on this source-held-out evaluation by any method, but the simple factor-2 anchor had the best observed score and highest observed coverage.

## Interpretation and literature boundary

The previous regression result was not enough: predicting a source's average tail-score rank does not estimate its upper tail accurately enough to set a local interval threshold. This is the central negative result of the probe. It also shows why a metric-only correlation gain must not be called interval efficiency.

The construction is a local/conditional calibration baseline, not a new conformal principle. Localized conformal methods already localize calibration around test covariates, conditional-coverage methods formalize restricted subgroup/shift classes, and a 2026 spatial conformal paper combines local quantile regression with conformal calibration ([Guan 2023](https://academic.oup.com/biomet/article/110/1/33/6647831), [Gibbs et al. 2023](https://arxiv.org/abs/2305.12616), [Jiang & Xie, ICML 2026](https://proceedings.mlr.press/v306/jiang26at.html)). Their assumptions and asymptotic regimes do not establish transfer for this ten-source CIF setting. The focused novelty boundary is recorded in `V50_CONDITIONAL_RISK_LITERATURE_NOTE.md`.

No conditional q, source weighting, or local calibrator is frozen. No genuinely unseen region is opened. A more credible paper contribution remains an information-budget benchmark and identifiability study that measures when monthly totals, full fuel composition, time-local operational information, and local hourly responses can or cannot support reliable hourly risk.

## Reproduction

```bash
../tran-stable/.venv/bin/python scripts/run_v50_coarse_mix_conditional_interval.py
```

The run writes 280 source-fold method rows and per-source summaries to `results/disjoint-calibration-v50-coarse-mix-conditional-interval/`, and records input/protocol/script/output hashes.

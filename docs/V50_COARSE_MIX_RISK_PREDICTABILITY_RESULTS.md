# V50 low-frequency mix → hourly-risk predictability probe

Date: 2026-10-27. This source-held-out diagnostic asks whether the low-frequency fuel statistic already available in the public monthly context predicts normalized hourly interval risk. It reuses only the sealed V50 external calibration source package. It does not alter V50 or evaluate an unseen region.

## Protocol and audit

The probe was registered in `V50_COARSE_MIX_RISK_PREDICTABILITY_PROTOCOL.md` before reading the coarse-context values. It opened 70 sealed outer-by-external prediction packages and ten external label packages, with zero target-package or target-score reads and zero point/risk refits. It fitted only the predeclared one-feature source-held-out diagnostic regressions. Seven outer-specific outcomes were aggregated to one outcome per source; the ten regions, not the hours or repeated outer functions, are the units.

The predictor is the mean of the latest three legally released monthly contexts' `POS` share across that source's July–December query dates. The primary outcome is the median across seven outer functions of the source-specific p90 of normalized hourly scores for the fixed `Scale-anchor + MCStd`, B0, 90-day setup. p99 is secondary.

## Result

The one-feature POS model did **not** improve source-held-out p90 prediction over the null predictor that uses the mean risk of the other nine regions:

| Outcome | Null mean | Monthly POS linear model |
|---|---:|---:|
| p90 MAE | 2.009 | 2.188 |
| p90 RMSE | 2.154 | 2.347 |
| p99 MAE | 2.682 | 2.941 |
| p99 RMSE | 2.831 | 3.117 |

The POS regression slope was positive in all ten folds, and the source-level Pearson correlation with p90 was 0.26, but the held-source errors were worse. High-risk sources such as NWMT, WACM and SC were not adequately predicted by their mean POS share. With only ten previously used regions, neither the slope consistency nor the correlation supports a generalization claim.

## Meaning for the research direction

The result rejects a tempting shortcut: a single local monthly positive-emission share is not enough to predict the residual hourly tail risk well enough to justify narrower intervals in a new region. This does not reject all monthly information. Renewable mix, within-month changes, reporting age, annual factors, or truly time-resolved public operational information remain separate hypotheses and would need preregistered, source-level evaluation against the same-budget simple and strong baselines.

Together with the previous source-influence probe, the current evidence says: pooled calibration sees real regional heterogeneity, while a single coarse POS summary does not explain it out of source. So the next valuable mechanism is likely to need **time-local or fuel-specific information tied to schedule/support**, not just more accurate estimation of one average mix. This is a design implication, not yet a discovered algorithm or a claim that such information is sufficient.

No method is frozen; no genuinely unseen region is opened.

## Reproduction

```bash
../tran-stable/.venv/bin/python scripts/probe_v50_coarse_mix_risk_predictability.py
```

The script refuses to overwrite its output directory and records source, held-out prediction, summary, and input/script hashes in `results/disjoint-calibration-v50-coarse-mix-probe/`.

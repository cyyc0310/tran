# V50 full monthly fuel mix → hourly-risk probe

Date: 2026-10-27. This source-held-out information-value probe follows the negative POS-only result. It uses the same ten external 2024 calibration regions and the same precomputed source-level hourly-risk outcomes; it does not alter V50 or open target/unseen regions.

## Protocol and data discipline

The protocol was registered in `V50_MONTHLY_FULL_MIX_RISK_PROTOCOL.md` before reading the full monthly mix values. The script verified the two monthly CSV hashes against the frozen V50 manifest, enforced full-month release timestamps, and selected only the three latest legal contexts at each query date. It opened no target-role prediction, 2025 response, PACE/LGEE, or unseen-region data. No point or risk production models were refit. Only ten folds of a one-feature source-level diagnostic regression were fit.

The feature is the project fixed-EF weighted average of the eight monthly generation shares, averaged over the three legal contexts and the July–December query dates. It is a generation-side monthly proxy, not observed CIF and not a certified China emission factor.

## Results

| Held-source metric | Constant mean | Full monthly mix proxy | POS-only proxy |
|---|---:|---:|---:|
| p90 MAE | 2.009 | **1.498** | 2.188 |
| p90 RMSE | 2.154 | **1.767** | 2.347 |
| p99 MAE | 2.682 | **2.064** | 2.941 |
| p99 RMSE | 2.831 | **2.414** | 3.117 |

Full monthly mix reduced absolute p90 and p99 errors in 7 of 10 held-source folds; fold slopes were positive in all ten. However, held predictions remained materially wrong for NWMT and WACM, where observed p90 scores were 4.95 and 5.81 while predictions were 2.76 and 3.31. SEC was also overpredicted (3.80 vs 0.81). These are not small-sample confidence bounds; they are descriptive errors over ten correlated development regions.

The earlier conditional within-POS factor probe did not produce an output because its preregistered support check failed: at least one required monthly context had zero POS mass, so a POS-conditional intensity was undefined. No fill value was used. The full-mixture proxy is defined at zero POS and is a new registered probe, not a silent repair of that failed design.

## Interpretation

The result is a promising *directional* signal that detailed monthly fuel mix can explain part of source-level risk better than POS mass alone. It does not establish a calibration method, interval coverage, interval-width benefit, or generalization to China. In particular, reducing an aggregate score-prediction error does not guarantee that a conditioned conformal threshold will improve coverage and sharpness.

Before calling this an innovation, the next experiment must compare a coarse-mix conditional risk rule against equal-region/simple anchoring and strong conditional-quantile/heteroscedastic baselines using identical information, with source-level held-out coverage and interval score. The protocol must also avoid using the held source's hourly labels to choose its own threshold. No method is frozen and no unseen target is opened.

## Reproduction

```bash
../tran-stable/.venv/bin/python scripts/probe_v50_full_monthly_mix_risk.py
```

The script records all input, source outcome, protocol, and output hashes in `results/disjoint-calibration-v50-full-mix-probe/` and refuses to overwrite an existing output directory.

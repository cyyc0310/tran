# V50 daily as-of mix level correction: seasonal-control result

Run date: 2026-10-27. The frozen protocol is [V50_DAILY_ASOF_SEASONAL_CONTROL_PROTOCOL.md](V50_DAILY_ASOF_SEASONAL_CONTROL_PROTOCOL.md). This is a source-held-out diagnostic on the same ten prior development sources; it does not open any target/unseen source.

## Results

The calendar-only control did not reproduce the mix correction. Across seven outer functions × ten held sources, hourly CRPS was 125.14 for no shift, 112.80 for an equal-source global shift, 113.28 for annual sine/cosine season-only, 71.43 for daily-as-of mix-only, and 71.62 for mix plus season. The mix-only model beat season-only CRPS on 8/10 sources. Its overall daily level MAE was 93.47 versus 142.33 for season-only. Thus the earlier mean gain is not explained by this simple shared annual calendar basis.

Adding the seasonal terms to the mix model made almost no macro difference: hourly CRPS changed from 71.43 to 71.62, daily-level MAE from 93.47 to 93.76, coverage from 95.27% to 95.24%, width from 478.64 to 478.89, and interval score from 510.42 to 510.17. The fuel-mix feature carries the useful signal in this tested basis; the seasonal terms add no material predictive value.

## Remaining failure modes

This control does not resolve the principal reliability issues. SEC coverage remains 75.18%; SCL remains dominated by support clipping. The leave-one-source summary shows mix-only hourly CRPS worse than the no-shift anchor for AVA, BANC, SEC, and SWPP, even though its macro score is much better. This is descriptive evidence of a heterogeneous correction, not a calibrated universal transfer rule.

## Decision

The seasonal negative control strengthens the scientific case that low-frequency fuel composition contains information about cross-region daily level error beyond a simple calendar basis. It is a useful incremental result, but it does **not** meet the agreed method gate because source-level coverage has a severe failure and one source has extreme clipping. No method is frozen and no unseen source is opened. The next optimization should constrain the magnitude of the learned offset using training-source-only shrinkage, with all shrinkage strengths reported and no post-hoc best-lambda claim; it must preserve per-source coverage and avoid the SCL clipping pathology before confirmation.

## Artifacts

- Script: `scripts/probe_v50_daily_asof_seasonal_control.py`
- Output: `results/disjoint-calibration-v50-daily-asof-seasonal-control/`
- Summary includes protocol, code, seal, input, and output hashes.

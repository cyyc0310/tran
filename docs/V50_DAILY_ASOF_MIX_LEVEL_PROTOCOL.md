# V50 daily as-of monthly mix for level correction

Registered 2026-10-27 as a temporal-integrity follow-up to the source-window level-bias probe. The prior probe used a full-window mean of per-date legal monthly features; this round tests a strictly date-local version so a July prediction cannot use a mix context first available later in the evaluation window.

## Fixed feature and source-held-out model

- Rebuild one feature for each external region and each registered V50 calibration query date: the mean fixed-EF intensity proxy of the three latest full-calendar fuel-mix records whose period ended and whose publication time was both no later than that query date. Use only the two hash-bound V50 external monthly CSV inputs and all eight recorded fuel shares.
- Use the same 90-day eligible hourly labels and seven frozen outer point functions. Hold out one entire source; no held-source labels enter either correction.
- For each outer, compute each training source-day response as the mean of `truth − Scale-anchor raw-path mean` over the 24 hours. Fit an intercept-plus-one-feature weighted least squares model on the other nine sources, with each source assigned equal total weight (each eligible day weight is inverse source day count). Predict a separate level shift for each held-source query date from only that date's legal feature.
- Simple comparison: a constant correction equal to the equal-source mean training bias. Additional controls: no shift and the fixed raw anchor factor-2 interval.
- Apply the shift equally to all 128 hourly paths of that day and clip to [0,1000]. Do not change intra-day deviations, refit the point/risk model, or recalibrate the factor-2 interval.

## Evaluation and limitations

Report per-source and macro daily-level MAE/bias, hourly and daily-mean CRPS, factor-2 coverage/width/interval score, and clipping frequency. The ten sources are the units; daily labels train a fixed low-dimensional regression but are not independent inferential samples. All ten sources were previously used in development. No significance, China transfer, coverage guarantee, method novelty, or unseen-region access may be claimed from this probe.

Verify the frozen V50 seals, all 70 external prediction hashes, ten label hashes, frozen monthly-input hashes, and lag support. Do not open target-role packages, target scoring files, proper archives, PACE/LGEE, 2025 responses, or unseen sources.

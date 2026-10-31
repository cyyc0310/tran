# V50 calibration-source influence diagnostic

Date: 2026-10-27. This is a read-only descriptive follow-up to the completed V50 formal run. It does not change V50 predictions, thresholds, or scores and does not qualify as target validation.

## Registered scope and audit

The protocol was fixed in `V50_CALIBRATION_SOURCE_DIAGNOSTIC_PROTOCOL.md` before running the diagnostic. The implementation opened exactly the 70 sealed outer-by-external prediction packages and ten external calibration-label packages. It opened no target prediction package, target response/scoring file, proper-fold archive, model bundle, PACE/LGEE data, or unseen-region data; it performed zero fits.

For each of the 1,260 registered outer × base × risk-method × lag × calibration-rule configurations, hourly normalized scores were independently recomputed from sealed bounds, scale, and external truth. The resulting 12,600 source/configuration rows were generated only after all 1,260 thresholds matched the sealed official values exactly (maximum absolute difference 0). The full source table and hashes are in `results/disjoint-calibration-v50-source-diagnostic/`.

## Results

- **Equal-region thresholds are usually insensitive to deleting one source, but a few high-risk sources matter.** Across leave-one-source-out configurations, the median threshold change was about +0.5 normalized-score units for every lag; its mean was near zero. Removing SC lowered the threshold in 80.8% of configurations (median change −4.33, most negative −85.31); removing NWMT did so in 76.5% (median −2.46, minimum −48.84), and removing WACM in 76.3% (median −1.44, minimum −45.19). Most other single-source deletions slightly raised the threshold. These are source influence diagnostics, not grounds for excluding those regions.
- **The HCP leave-one-source-out jumps mostly reflect the finite-source correction.** With ten sources the finite level is .99; after deletion, the registered HCP formula for nine sources requires level 1.0, so the threshold becomes the maximum remaining score. The median increase of roughly 58–67 units therefore cannot be interpreted as evidence that the deleted source itself caused the original threshold. Only 0.48% of HCP leave-one-source cases reduced the threshold.
- **Hourly score tails differ substantially across calibration regions.** Averaged over the fixed configurations, the local p99 was 246.5 for SC, 223.6 for NWMT, and 220.7 for WACM, compared with 2.5 for SCL and 39.8 for BANC. At the official threshold, average local exceedance was about 15.9% for SC, 15.7% for NWMT, and 11.0% for WACM, versus below 0.2% for AVA, BANC, and SCL. Thus a pooled marginal threshold can over-cover some calibration regions while under-covering others.
- **Changing 90/30/7-day support did little to the equal-region result.** Its median deletion effect stayed near +0.5 across all three lags. The HCP pattern also stayed similar because its finite-sample correction, not the lag choice, dominates the leave-one-source comparison.

## Research implication

This does not reveal a new conformal rule or establish a defensible source-removal policy. It does identify a concrete unresolved mechanism: hourly risk is strongly region-heterogeneous even after normalization, so pooled thresholds cannot deliver region-conditional reliability. A potentially valuable next experiment is to test whether the legally available low-frequency generation mix in the intended deployment setting predicts **risk heterogeneity** across source regions, using a source-held-out evaluation and the same information budget for simple and strong baselines. This must be evaluated at the region level; treating thousands of hours from ten regions as independent would create false precision. If low-frequency mix carries no out-of-region signal, it cannot justify narrower intervals in an unseen Chinese region.

No method is frozen and no unseen target is opened. V50's main finding remains: the calibrated intervals did not show a stable efficiency gain over simple anchoring, and current evidence is all from repeatedly used development regions.

## Reproduction

Run from the repository root with the project environment:

```bash
../tran-stable/.venv/bin/python scripts/diagnose_v50_calibration_source_influence.py
```

The script refuses to overwrite an existing diagnostic directory and verifies the formal seals, file hashes, fixed source grid, lag counts, and every reconstructed official threshold before writing results. The protocol, source script, source table, and summary hashes are recorded in `summary.json`.

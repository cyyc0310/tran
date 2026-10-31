# V50 daily as-of monthly mix level correction: source-held-out result

Run date: 2026-10-27. Frozen protocol: [V50_DAILY_ASOF_MIX_LEVEL_PROTOCOL.md](V50_DAILY_ASOF_MIX_LEVEL_PROTOCOL.md). This is a follow-up to the window-average diagnostic in [V50_MONTHLY_MIX_LEVEL_CORRECTION_PROTOCOL.md](V50_MONTHLY_MIX_LEVEL_CORRECTION_PROTOCOL.md).

## Result

The stricter query-date feature retained a sizeable pooled signal. Across ten previously used external sources and seven fixed outer functions, the daily-as-of monthly-mix linear correction reduced source-macro hourly CRPS from 125.14 (no shift) and 112.80 (equal-source global bias shift) to 71.43. Daily-mean CRPS was 123.50 / 111.47 / 68.02, respectively. The correction reduced the absolute source-level bias MAE from 138.41 / 136.56 to 88.93; it beat the global correction on 7/10 held sources.

Against the no-shift raw factor-2 interval, the mix correction had macro coverage 95.27% vs 88.50%, mean width 478.64 vs 527.98, and interval score 510.42 vs 712.16. It improved hourly CRPS on 6/10 held sources, interval score on 7/10, and width on 6/10. Against the simple global shift it improved hourly CRPS, interval score, and width on 8/10, 8/10, and 7/10 sources.

A post-hoc influence check on the ten held-source summaries shows the pooled hourly CRPS reduction is not caused by one source alone: removing any one source leaves a negative mean CRPS difference (mix minus no-shift, range −35.47 to −66.28). The median source-level CRPS difference is −30.94, but four of ten sources still worsen. Coverage tells a less favorable story: its median source-level difference is exactly zero, and only four sources improve coverage. This confirms a broad average CRPS signal alongside unresolved region-specific calibration failure; source deletions are descriptive sensitivity checks, not additional independent validation.

This pooled gain is not a reliable all-region result. The held-source coverage ranged from 75.18% (SEC) to 100%; SEC is a material undercoverage failure. SCL's mix-corrected paths were clipped to [0,1000] for 88.88% of particle-hours, making its apparently excellent scores/coverage physically and statistically suspect. The correction worsened hourly CRPS for AVA, BANC, JEA, and SCL versus the uncorrected anchor. The aggregate improvement is therefore heterogeneous and partly depends on large baseline errors in a few sources; it does not meet a robust per-source coverage criterion.

## Validity and limitations

- The feature was rebuilt for each query date from the three most recent complete monthly eight-fuel records with `period_end` and `available_at` no later than that date. The 70 prediction packages and ten external label packages matched the frozen V50 seals and hashes. No target-role prediction/score, PACE/LGEE, or unseen source was opened; no point or risk model was refit.
- The method is an equal-source-weighted, leave-one-source-out intercept-plus-one-feature regression of daily mean residuals. It translates monthly mix into an additive daily level shift, preserving within-day path deviations before clipping. Comparisons were no correction, a simple global source bias shift, and the same raw factor-2 interval. This was not a complete same-budget comparison against the V50 heteroscedastic/conditional-quantile arms.
- The feature uses the frozen external proxy tables and fixed project emission factors. Timestamp legality is verified against those tables, but this is not independent proof that each timestamp equals the original public release time. The fixed factors are not a China-validated emissions accounting system.
- All ten sources have been used in prior development. Seven outer functions and daily rows do not increase the independent source count. Treat these as descriptive source-held-out evidence, not a confirmatory trial or coverage guarantee.
- The earlier window-average experiment showed CRPS 67.98, coverage 95.82%, width 477.43, interval score 503.25, and 11.0% clipping, but reused an average feature spanning the full evaluation window. It is retained as a look-ahead diagnostic only; the valid daily-as-of result is the one reported above.

## Decision

This is a stronger, temporally valid *signal* that monthly fuel composition may help correct systematic daily level bias across some regions. It is not yet a breakthrough: one source misses coverage badly, another is dominated by clipping, the signal is heterogeneous, and no truly unseen source has been evaluated. Before any method freeze, the next useful round is a pre-registered stress test of the correction under alternative fixed-factor/feature parameterizations and a stronger same-budget baseline comparison, with source-level worst-case coverage and clipping treated as hard diagnostics. Any positive method must then be tested on a genuinely unseen source using a frozen protocol.

## Artifacts

- Output directory: `results/disjoint-calibration-v50-daily-asof-level/`
- Script: `scripts/probe_v50_daily_asof_mix_level.py`
- Run summary records the frozen input, seal, protocol, script, and output hashes.

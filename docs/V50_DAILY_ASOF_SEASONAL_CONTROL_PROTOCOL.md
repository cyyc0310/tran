# V50 daily-as-of monthly-mix correction: seasonal control

Registered 2026-10-27 before this follow-up computation. This tests whether the source-held-out monthly-mix level-correction result is merely a shared calendar-season effect.

## Fixed design

- Reuse only the frozen V50 prediction/calibration seals and the two hash-bound external monthly CSVs already named in `V50_DAILY_ASOF_MIX_LEVEL_PROTOCOL.md`. Verify all hashes again. Rebuild the same three-latest-legally-released-month fixed-EF feature at each query date; do not average across future dates.
- Reuse the same seven fixed outer functions, ten external sources, 90-day eligible labels, Scale-anchor raw paths, [0,1000] clipping rule, and factor-2 interval. Do not read target-role or unseen-source artifacts and do not refit point/risk models.
- For each outer and held source, train equal-source-weighted OLS on the other nine sources' daily residual responses. Compare five prespecified offsets: zero; equal-source global constant; mix-only (intercept + daily mix EF proxy); season-only (intercept + annual sine/cosine of query day, period 365.25 days); mix+season (intercept + mix proxy + the same sine/cosine terms). Each source has total training weight one, distributed uniformly across its eligible days.
- Apply each estimated offset to all 128 hourly paths of the corresponding held-source day. Preserve original factor-2 interval construction. Report source-level and source-macro hourly/daily CRPS, daily-level MAE/bias, coverage, width, interval score, and clipping. No significance tests; source is the independent unit.

## Interpretation boundary

This is a prespecified diagnostic control on ten previously used development sources. It can show whether the monthly feature adds descriptive source-held-out value beyond a simple calendar basis in this dataset. It cannot establish China transfer, guarantee coverage, prove algorithmic novelty, or trigger access to unseen sources. The fixed EF proxy and monthly release timestamps retain the limitations recorded in the prior result note.

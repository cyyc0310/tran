# V394: all-province release-lag composition-state audit

Date: 2026-10-31. This extends the official NBS provincial generation snapshot from five deliberately selected provinces to all 31 mainland provincial-level regions. The question remains limited to whether a public, time-valid monthly mix identifies a *coarse mean-support threshold state*. It is not an hourly CIF forecast evaluation.

## Data snapshot and audit

- Official NBS JSON API, monthly current-value series from 2018-01 through 2026-09 for total, thermal, hydro, nuclear, wind, and solar generation.
- 31 regions × 105 requested month codes × 6 series = 19,530 source rows. The API snapshot contains 3,563 missing current values. Missing values remain missing; cumulative values were not differenced to create monthly estimates.
- The extractor captured a source response for every region/technology request. Exact payloads, response hashes, and API metadata are in `results/v394-nbs-all-province-long-panel-r1/manifest.json` and `raw/`.
- All 3,150 rows for the five-province V392 overlap match V392 exactly, including missingness. The Shanghai user files' 33 current thermal/wind/solar values already match the official API exactly in V392; official hydro and nuclear are zero in 81 observed Shanghai months and remain missing in 24 months.
- Of 15,451 nonmissing official values overlapping the project V320 transcription, 13,182 match exactly and all 15,451 lie within half of the official 10 GWh display step and match after half-up rounding to that step. The 2,269 raw non-exact values are solar; the largest difference is 5 GWh. This is a precision consistency check against a transcription from the same statistical system, not independent-source validation.

The API gives a current historical series rather than as-published vintage snapshots. Release timing supports the month-start convention of using `t-2`, but historical revisions and original vintage values are unavailable.

## State and baselines

For local thermal share `s`, use the V392 certificate `400*s <= 300 < 1000*s`. The state is one when the assumed thermal mean-support interval crosses 300 g/kWh, which corresponds to `0.30 < s <= 0.75`; otherwise the mean-support interval lies wholly on one side. Thermal EF support is `[400,1000]` g/kWh and nonthermal operational EF is set to zero for this diagnostic.

At the start of target month `t`, use each province's local `t-2` mix as the local feature. Comparators are (1) the pooled 31-region mix from `t-2`, (2) a leave-one-province-out pooled mix from the other 30 regions at `t-2`, and (3) province-specific static state fitted only on target states through 2022. The static state is a time-forward comparator, not a cold-start region baseline.

## Results

On the common panel of 2,139 valid region-months, local `t-2` state has 89.1% accuracy and 88.8% balanced accuracy (234 errors). The leave-one-province-out pool has 38.8% accuracy and 50.2% balanced accuracy; it labels nearly every target as crossing and misses clear-above and clear-below regimes. The all-province pooled mix is nearly identical at 39.0% accuracy and 50.4% balanced accuracy.

In the forward period 2023-01 through 2026-09, the common sample has 899 region-months. Local `t-2` achieves 87.7% accuracy / 87.8% balanced accuracy, compared with 45.6% / 50.0% for the leave-one-out pool and 81.3% / 80.0% for the province-static state fitted through 2022. This is promising evidence that a local public monthly feature updates a stale static regional classification and resolves heterogeneity that a national pool erases.

The effect is not uniform. In the all-period local `t-2` rows, agreement ranges from 55.1% in Zhejiang and 58.0% in Chongqing to 100% in several provinces with stable threshold regimes. Zhejiang, Chongqing, Xinjiang, Guangdong, Heilongjiang, Hebei, and Jilin account for most misses. In the 2023+ forward sample, local lag-2 errors are concentrated in a handful of provinces including Chongqing, Zhejiang, Hebei, Inner Mongolia, and Heilongjiang. Full regional and year tables are included in the results directory.

## Scientific interpretation

This is a real gain for **threshold-state identification** under a modest public-information budget: at month-start, the local mix classifies the eventual mean-support state much better than the lagged nationwide aggregate. It also makes the limitations measurable: roughly one in nine states are still wrong, error rates vary by region, and the cold-start obstacle is not solved by simply assuming a stable province regime.

It is **not yet a CIF algorithm breakthrough**. Both predictor and target are composition-derived mean-support states. There are no hourly China CIF labels, no hourly `K` set, no probability coverage, and no interval width/score evaluation. The historic API lacks vintage snapshots. The next decisive experiment must add an independently measured hourly CIF target (or an auditable hourly generation/emissions chain) and evaluate a frozen local-composition predictor against national anchors, static regional priors, and standard dynamic baselines on held-out provinces with coverage and interval efficiency reported together.

## Reproduction

```bash
python scripts/v394_nbs_all_province_monthly_snapshot.py --start 201801MM --stop 202609MM --output results/v394-nbs-all-province-long-panel-r1
python scripts/analyze_v394_all_province_lopo.py
```

Key artifacts:

- Official snapshot: `results/v394-nbs-all-province-long-panel-r1/`
- Cross-province metrics: `results/v394-all-province-lopo-r1/audit_summary.json`
- Province detail: `results/v394-all-province-lopo-r1/metrics_by_province.csv`
- Forward comparison: `results/v394-all-province-lopo-r1/forward_holdout_metrics.csv`
- NBS/V320 comparison: `results/v394-all-province-lopo-r1/official_vs_oreo_crosscheck.csv`
- Analysis scripts: `scripts/v394_nbs_all_province_monthly_snapshot.py`, `scripts/analyze_v394_all_province_lopo.py`

# V410: release-lagged coarse mix under China hourly scenario stress

> Provenance correction (2026-10-31): all five scenario labels were previously opened in V194–V200. S2–S5 are excluded from this runner's fitting/calibration, but are not project-wide unseen scenarios. Treat V410–V411 as exploratory development replication; see [correction](V410_V411_PROVENANCE_CORRECTION.md).

Date: 2026-10-31. Protocol: [V410](V410_CHINA_SIMULATED_CIF_SCENARIO_STRESS_PROTOCOL.md). Runner: [`v410_china_simulated_cif_scenario_stress.py`](../scripts/v410_china_simulated_cif_scenario_stress.py). Raw source snapshot and outputs: [`results/v410-china-simulated-hourly-cif-stress-r1/`](../results/v410-china-simulated-hourly-cif-stress-r1/).

## Data and scope

The Figshare item contains 2025–2060 projected provincial factors from a planning/dispatch simulation and carbon-flow model, with five planning scenarios; version 3 is CC BY 4.0. The 2025 sheets have 8,760 hourly rows and 31 provincial columns. The publication says the projection uses generation schedules and power flows from a two-layer planning/operation model and then computes provincial factors through a carbon-emission-flow model; its core planning/operation code is not shared. We therefore treat the file as a **scenario stress reference**, not observed hourly CIF or a reproducible independent model. See the [data article](https://www.nature.com/articles/s41597-026-07272-6) and [Figshare record](https://figshare.com/articles/dataset/Hourly_carbon_emission_factors_of_31_provinces_and_cities_in_China_from_2025_to_2060/28953545).

Scenario S1 was used for development/calibration. S2–S5 were held out as load-growth/carbon-budget stress scenarios. The NBS feature at target month `t` was the five-category generation mix from `t−2`; only May–December were complete for all 31 provinces. This uses an archived current historical NBS snapshot, not a release-vintage replay. NBS monthly generation is production-side and above-scale-enterprise scoped, whereas the scenario target is a provincial network-flow factor. This is a useful mismatch stress, not a matched-estimand validation.

## Frozen comparison

Each target province was excluded from S1 source fitting and calibration. The methods were a source-only hourly profile, a fixed `700 × lagged local thermal share` level anchor, and a source-trained Ridge using the five lagged technology shares plus calendar harmonics. Ninety-percent monthwise simultaneous intervals used leave-one-source-province-out residuals, with the target province excluded from calibration. All 31 provinces, eight months, and four stress scenarios were retained: 248 province-months per scenario and method.

## Results

| Scenario | Method | Region-equal point MAE (gCO₂/kWh) | RMSE | Mean hourly interval width | Simultaneous month coverage | Worst-province coverage |
|---|---|---:|---:|---:|---:|---:|
| S2 | Source hourly profile | 202.3 | 214.8 | 962.0 | 91.9% | 0% |
| S2 | Local thermal anchor | 138.5 | 155.6 | 945.0 | 90.7% | 0% |
| S2 | Coarse-mix Ridge | 141.5 | 155.9 | 938.4 | 91.5% | 0% |

S3–S5 show the same pattern: the local level anchor and coarse Ridge substantially reduce point error versus a source-only hourly profile, but the Ridge does not improve point MAE over the simpler thermal-share anchor. Mean intervals stay about 938–962 gCO₂/kWh wide. Beijing has zero simultaneous month coverage in all eight evaluated months for both the anchor and Ridge; source-only transfer also has provinces with zero coverage. Thus the near-90% region-mean coverage hides severe regional failure.

For the 300/500/700 gCO₂/kWh thresholds, all methods' induced K-count intervals contain every observed simulated K, but the sets are broad: mean width is about 735 hours at τ=300, 692–735 hours at τ=500, and 607–730 hours at τ=700. At τ=300 there are no positive or negative certificates. At τ=700 and the 1% event target, the Ridge rules out a tail in 12.5% of province-months, while 87.5% abstain; there are no positive certificates. The anchor rules out 13.7%, but with a wider K set. These are weak selective frontiers rather than a practical high-risk predictor.

## Exploratory level–shape follow-up (V411)

After inspecting V410, a post-hoc additive decomposition was tried: anchor the target hourly level at `700 × local thermal share` and transfer the mean source hourly anomaly after subtracting each source's monthly mean. This is recorded separately as discovery-only in [`v411_level_shape_scenario_exploratory.py`](../scripts/v411_level_shape_scenario_exploratory.py) and [`results/v411-level-shape-scenario-exploratory-r1/`](../results/v411-level-shape-scenario-exploratory-r1/).

Across S2–S5, it lowers point MAE to 133.5–134.5 gCO₂/kWh and RMSE to 147.6–148.6, versus the fixed local anchor's MAE 138.3–139.1 and RMSE 155.3–156.0. However, mean simultaneous interval width remains 941.5 gCO₂/kWh; worst-province coverage is still 0–12.5%; τ=700 K-set width is 632 hours, with only 9.7% negative certificates and 90.3% abstention. This is a modest shape-transfer signal, not interval-efficiency or coverage success, and the follow-up is post-hoc on already opened S2–S5 labels.

## Research conclusion

This stress test does **not** establish a forecasting breakthrough. It clarifies the failure mechanism: local thermal share carries a useful low-frequency level signal, while simple regional shape transfer yields at most a small point-error gain; neither resolves provincial carbon-flow/exchange differences or hourly risk, and the prediction intervals remain nearly full-range. In particular, production-side NBS composition cannot by itself identify a provincial network-flow factor when the target includes imported electricity.

The most promising next contribution is therefore an **estimand- and information-budget benchmark** that separates (1) generation-side monthly mean support, (2) consumption/network-flow exposure, and (3) within-month hourly tail risk. A further model should be pursued only if a legal, target-matched local field—e.g. monthly receipt/source-carbon exposure—or a real high-frequency label can reduce abstention without sacrificing regional coverage. The simulation set can serve as an explicitly labeled stress suite, not as the primary empirical validation.

## Reproduction

Run `python scripts/v410_china_simulated_cif_scenario_stress.py`. The output `summary.json` records hashes of the V410 protocol, runner, source xlsx files, Figshare metadata, NBS snapshot, cohort size, and calibration rule. Detailed outputs include `region_month_metrics.csv`, `tail_count_intervals.csv`, `summary_by_scenario_method.csv`, `tail_frontier.csv`, and `calibration_region_month_scores.csv` (all 30 region-level calibration scores per target province/method/month).

# 2022 annual capacity paired US↔GB development run

Status: `DEVELOPMENT_CAPACITY_STAT_YEAR_2022_SCORED_ON_PREVIOUSLY_VIEWED_2024_TARGETS`. The 2024 target was already evaluated in an earlier project run. This package recomputes forecasts under a 2022 coal/gas capacity feature budget, preserves the original 2020-capacity forecasts as the paired comparator, then scores both on the same existing target response hours.

Source training and calibration years are 2022 and 2023. US 2022 capacity uses the archived original EIA-860 ZIP; GB 2022 capacity is taken from the 2026 revised DUKES Table 5.7 because the original 2023 supporting workbook was not recoverable. It is a current-revised statistic-year comparison, not a strict historical as-issued replay. UK includes Northern Ireland and is a GB proxy.

`pre_score_seal.json` holds the forecast hashes and code/input hashes captured before scoring. `target_scoring_hashes.json` identifies the already-viewed target response files. `paired_region_metrics.csv` contains per-region paired metrics by capacity year; `monthly_metrics.csv` preserves month-level scores. The complete report explains interpretation and limits.

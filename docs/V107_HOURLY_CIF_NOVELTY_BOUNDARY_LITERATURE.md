# V107: novelty boundary from hourly emissions-factor literature

## Literature check

1. The IEA already publishes monthly and quarterly generation CO₂ intensities for 52 countries through 2024 Q2. Its documented construction combines monthly generation with annual fuel-specific generation factors and annual scaling. Therefore, building a monthly generation-based intensity from monthly fuel totals and annual factors is established practice; it is not a defensible novelty claim. The IEA procedure stops at monthly/quarterly aggregation and does not identify within-month hourly K-tail risk.
2. Short-term generation-based CO₂-factor forecasting is also established: the 2024 German study computes hourly factors from hourly generation mix and compares parametric/nonparametric models with walk-forward splits to forecast 24 hours ahead. A new paper cannot claim novelty merely for an hourly CIF forecast model or for standard time-series baselines.
3. Monthly/hourly marginal-emissions-factor modeling from demand, renewable generation, and emissions is an adjacent but different estimand: marginal response to load changes is not the same as average generation-based CIF. The distinction must be stated explicitly.
4. A 2026 Applied Energy study compares annual, seasonal, hour-of-day, season-hour, month-hour, and hourly factors against hourly reference factors for building emissions accounting. It finds time resolution matters and month-hour/season-hour summaries can be effective for that accounting task. This reinforces that temporal coarsening is a known problem, while not resolving partial identification of rare hourly threshold counts from local aggregate statistics.
5. Recent China hourly-factor work located in the search used seven months of China Southern Power Grid data obtained under a confidential agreement. This suggests an hourly validation channel exists institutionally, but not that a public reusable panel is available. A real China claim needs an auditable permission/data contract or a reproducible public proxy with an explicit estimand.

## Consequence for this project

The strongest defensible distinction is not a new point predictor. It is an information-adequacy benchmark for **hourly average generation-based CIF tail risk**: given only target-month fuel/technology totals and lagged annual coarse capacities, compute the exact or conservatively robust identified set for the number of hours above a threshold; compare the width and selective decision frontier as the legally available local information budget increases; count unavailable cases; and show when the target is not identifiable. V104–V106 support the mechanism but also reveal high abstention. The benchmark claim becomes valuable only if it includes a theorem for the sharp set/robust envelope, preregistered cross-region validation, field-vintage and factor contracts, and at least one real hourly CIF evaluation source.

If the paper remains framed as forecasting an hourly **marginal** emissions factor, the current code and fixed-factor proxy do not support that claim. The pipeline computes a generation-weighted average from fuel output and assumed factors. Renaming this estimand is not a wording change; it requires different hourly labels and models.

## Sources

- IEA, [Monthly and Quarterly Emission Factors database documentation](https://www.iea.org/data-and-statistics/data-product/monthly-and-quarterly-emission-factors), and [technical methodology PDF](https://iea.blob.core.windows.net/assets/97160ad1-eb43-4317-a20f-082516d83b78/IEAMonthlyandQuarterlyEmissionFactors-documentation.pdf).
- 2024, [Short-term forecasting of German generation-based CO₂ emission factors](https://doi.org/10.1186/s42162-024-00303-9).
- 2023, [A data-driven framework for quantifying consumption-based monthly and hourly marginal emissions factors](https://doi.org/10.1016/j.jclepro.2023.136296).
- 2026, [Temporal resolution matters: evaluating carbon emission factors for accurate accounting in commercial buildings](https://doi.org/10.1016/j.apenergy.2026.127655).
- 2025, [Quantifying Weather’s Share in Dynamic Grid Emission Factors via SHAP](https://doi.org/10.3390/pr13113393).

This is a scoped literature check, not a systematic review. No priority claim is made for the identified-set contribution.

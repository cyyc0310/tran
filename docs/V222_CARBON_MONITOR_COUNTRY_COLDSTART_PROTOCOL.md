# V222 Carbon Monitor country-held-out daily reconstruction protocol

Date: 2026-10-30. Status: external secondary-data stress test; no independent metered-CIF validation.

## Question

Can month-after, category-resolved generation totals support cold-start daily CIF reconstruction in a country with no target-country daily/CIF history in the model? Does choosing source countries by the target's current-month mix or by its historical monthly fuel-mix seasonality make cross-country daily shapes transferable?

## Data and information contract

- Input: the public Carbon Monitor Power global daily generation CSV, 75 country/aggregate series, eight source categories, 2019-01-01 to 2026-10-26 in the downloaded snapshot. The downloaded file SHA-256 is recorded in `results/v222-carbon-monitor-country-coldstart-r4/metadata.json`.
- Model training and donor weekday profiles use 2019–2023 only. Target countries are removed from donor candidates. 2024–2025 are the outer target period.
- For each outer target month, the model receives the target's eight category generation totals for that month. This is post-month information; scores describe retrospective daily reconstruction, not a month-start forecast.
- A second donor rule uses only target-country monthly category shares from 2019–2023 to compare 12-month seasonal mix signatures. It does not use target-country daily/CIF history.
- The daily labels are mechanically computed from the same source categories with fixed factors Coal=1000, Gas=400, Oil=800, Other=400, and Nuclear/Hydro/Wind/Solar=0 gCO2/kWh. They are not independent observed CIF.
- Include a target country-month only when all eight categories are present for every calendar day and total generation is positive.

## Frozen methods

1. `constant`: apply the target month generation-weighted fixed-factor mean to every day.
2. `cif_shape_k10`: average the ten current-month composition-nearest donor weekday CIF profiles; rebase the profile to the known target monthly fixed-factor anchor.
3. `fuel_shape_k10`: average ten current-month composition-nearest donor fuel-specific weekday profiles, allocate the target's eight exact monthly category totals across target days, and recompute daily CIF.
4. `fuel_shape_season_k10`: same fuel-profile allocation as method 3, but rank donors by average Bray–Curtis distance between their 2019–2023 twelve-month fuel-mix signatures and the target's signature.

Main score: mean absolute daily error, first averaged over days and months within each country, then equally averaged across countries. Paired 95% intervals resample countries (20,000 draws). Daily rows are not treated as independent domains.

## Interpretation boundary

The data source is itself an estimation product. The [2023 Carbon Monitor Power paper](https://www.nature.com/articles/s41597-023-02094-2) describes China daily fuel generation using monthly CEC generation totals and daily coal-consumption patterns from eight coastal provinces; the [currently downloadable dataset](https://power.carbonmonitor.org/) is a living/revised release, so its present values may not match the paper's original processing exactly. The cross-country test is therefore evidence about this secondary product and transfer assumptions, not truth-level CIF accuracy. No interval coverage result is claimed in V222.

## Reproduction

- Script: `scripts/v222_carbon_monitor_country_coldstart.py`
- Final exploratory run: `results/v222-carbon-monitor-country-coldstart-r4/`
- Input URL: <https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=energy_global>

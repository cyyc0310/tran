# V222 Country-held-out daily CIF reconstruction: shape migration did not beat the month anchor

Date: 2026-10-30. Final exploratory run: r4.

## Result

The complete-case test includes **75 held-out countries/aggregates, 1,800 country-months, and both 2024 and 2025**. Macro-country daily MAE (gCO2/kWh) was:

| Method | Macro-country MAE | Paired difference vs constant (95% country bootstrap CI) |
|---|---:|---:|
| Monthly constant anchor | 14.983 | reference |
| Donor CIF weekday shape, K=10 | 15.103 | +0.120 [0.080, 0.168] |
| Donor fuel weekday shape, K=10 | 15.200 | +0.217 [0.146, 0.293] |
| Donor fuel shape, K=10, seasonal-history matching | 15.211 | +0.227 [0.145, 0.325] |

Positive differences mean worse MAE. The fuel-shape method also worsened MAE vs donor CIF shape by +0.096 [0.014, 0.181]. Seasonal matching changed little versus current-month matching (+0.011 [−0.021, 0.045]); it did not rescue the transfer.

Only 34.0% of country-months favored donor CIF shape over the constant anchor and 32.4% favored fuel-shape transfer. The country-month average within-month label standard deviation was 17.99 gCO2/kWh (median 12.53, 90th percentile 41.93), so daily variation exists, but the tested source weekday signal generally did not transfer.

**China-specific result:** across 24 held-out months in 2024–2025, constant-anchor MAE was 3.70 gCO2/kWh; donor CIF shape was 3.88, donor fuel-shape was 4.85, and historical-seasonality donor matching was 4.81. This is not independent Chinese validation: the daily labels come from the same Carbon Monitor estimated generation series.

## What this says about the proposed direction

1. Knowing all eight target-month generation margins identifies the generation-weighted monthly mean under the fixed-factor assumptions, but does not by itself identify daily variation.
2. Simple cross-country weekday-shape transfer is not supported by this benchmark. In particular, increasing the resolution of target monthly margins does not make borrowed daily shapes reliable.
3. Target historical monthly mix signatures, which match the user's stated public-data budget better than hourly histories, did not help choose a transferable profile.
4. The defensible research direction remains an **information sufficiency/value benchmark**: distinguish what monthly margins identify exactly from what remains unidentified, then test whether a proposed minimum local temporal observation can reduce risk. A model that automatically adds donor shapes can be worse than returning the identified monthly anchor.

## Limitations and next decision

- Carbon Monitor Power is an estimated secondary dataset. Its China methodology in the 2023 paper used daily coal-consumption patterns from eight coastal provinces to disaggregate monthly generation. It supplies daily values rather than independent hourly plant output; the CSV is a living/revised dataset.
- Fixed emission factors are deliberately shared with project experiments for a controlled index, not claimed as location-specific lifecycle factors.
- This run evaluates point MAE only. It does not establish interval coverage or a calibrated rejection rule.
- The target month totals are available after the month, so this is reconstruction, not prospective forecasting.

The seasonal-match failure is a stop signal for more K tuning on this same dataset. The next useful experiment should test information acquisition directly (for example, sparse target-day generation snapshots) against the monthly-anchor baseline, with a frozen source-only calibration protocol and explicit coverage/width scoring. Any claimed gain must then be replicated on an independent operational data source before a paper claim.

## Files

- Protocol: [V222 protocol](V222_CARBON_MONITOR_COUNTRY_COLDSTART_PROTOCOL.md)
- Script: [runner](../scripts/v222_carbon_monitor_country_coldstart.py)
- Artifacts: `results/v222-carbon-monitor-country-coldstart-r4/` (daily predictions, country/month scores, paired country bootstrap, metadata)

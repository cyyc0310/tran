# R6b weather effect by held-out target country

- Runner: `scripts/analyze_expanded_weather_country_deltas.py`.
- Input: `results/expanded-anchor-issued-r6b/configuration_metrics.csv` (SHA-256 `0daafd733c98adffe04681ea7dd2cc43ffc50bb1d2c4f7449940b692800f557e`).
- Output: `results/expanded-weather-country-diagnostic-r1/country_paired_deltas.csv` (SHA-256 `d0d5557f839727a4ca8335e01efe7ad2608897c49b47e4593c879e46b47df5d0`).
- Pairs use the frozen `paired_hgb_no_weather` and `paired_hgb_weather` configurations with `calibrated` risk, averaged over the three source-target configurations for each held-out country and budget.
- This is a post-hoc score diagnostic on previously viewed targets. It refits no models, reads no new labels, and makes no significance claim; three target countries are too few for a general cross-country reliability inference.
- Main finding: M+C aggregate weather gains are concentrated in DE; GB and US metrics are mixed. Under M, point MAE improves in all three countries, but probability scores and coverage remain poor in GB/US.

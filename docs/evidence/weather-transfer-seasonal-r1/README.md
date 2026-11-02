# Cross-country native weather pilot evidence

See [the result report](../../WEATHER_TRANSFER_SEASONAL_RESULTS.md) and [protocol](../../WEATHER_TRANSFER_SEASONAL_PROTOCOL.md).

The manifest binds frozen scores, all prediction bytes (gzip), code, calibration/model seals, a post-score labeled cohort and original point-weather inputs. `native_metadata.tar.gz` consolidates 588 exact native metadata files; its member hashes are in the manifest. Original `.idx` and raw GRIB are omitted, so these weather folders deliberately are not compact caches accepted by the production loader. The original full caches and fitted model files remain local.

`cohort_with_labels.csv.gz` contains held-out labels solely for evaluation. It was created after prediction sealing and was never read by the frozen forecaster. Only 624 unique target hours (26 region-days) exist; method/configuration rows are repeated evaluations. This sampled cohort cannot retrain the full-source-year extra-label reference.

Original copied documents retain their original relative-link context for byte provenance. Use the report link above for rendered reading. No novelty, target coverage guarantee, certified historical public availability, or China accuracy is claimed.

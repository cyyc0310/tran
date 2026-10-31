# V233 Frozen protocol: provincial shape inheritance in Carbon Monitor China

Date: 2026-10-30. Frozen before analyzing normalized Power-sector provincial profiles.

## Question

Does the currently downloaded Carbon Monitor China daily Power CO₂ series contain province-specific within-month daily shapes, or are most differences a province-month scale applied to a shared daily shape? This matters for whether province-held-out daily CEF evaluation on this product can provide independent evidence of cold-start temporal transfer.

## Input and scope

Download the current Carbon Monitor China CSV from the official site's download link, `https://datas.carbonmonitor.org/API/downloadFullDataset.php?source=carbon_china`, to `/tmp/carbonmonitor_china_v233_live.csv` (fields: state, date, sector, value). Restrict to sector `Power`, parse `DD/MM/YYYY`, and retain valid nonnegative values. Hash the file before computation. This is an exploratory product-lineage audit, not a forecast or CIF benchmark. The result is tied to this live snapshot; the official portal notes that values and methods can be revised.

## Frozen analysis

For each year-month, retain provinces with at least 90% of calendar days and align their common dates. Normalize each province's daily Power series to sum to one within that month. Report number of aligned province-month blocks/days, pairwise profile correlation, pairwise normalized L1 distance, maximum deviation from the cross-province mean daily profile, and the residual error from the rank-one model `emissions[p,d] = province_month_scale[p] × common_month_day_shape[d]`.

Also report complete-data coverage and missingness by province, month, and year. Do not infer hourly CIF or emissions accuracy from this shape diagnostic.

## Interpretation limits

- The product is an estimated, revisable daily emissions dataset, not independent direct measurement or hourly CIF.
- Near-identical within-month shapes would show limited independent spatial-temporal information in this product, not prove the underlying power system has identical provincial dynamics.
- This analysis does not reconstruct or criticize any paper's unpublished implementation; it only tests the public product snapshot's structure.

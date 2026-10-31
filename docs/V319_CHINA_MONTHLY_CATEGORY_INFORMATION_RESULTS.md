# V319 results: coarse monthly category fields retain identifiable information after correcting source assignment

Run date: 2026-10-30. Frozen protocol: [V319](V319_CHINA_MONTHLY_CATEGORY_INFORMATION_PROTOCOL.md). Full province-month output: `results/v319-china-monthly-category-information-r1/province_month_information_diameter.csv`; inputs and protocol are SHA-256 bound in `summary.json`.

## Correction to V315/V316

An audit found that V315/V316 assigned each incoming source-technology flow the **receiver's** technology factor. The correct source attribution uses the **origin province's** factor. This changes the LP objective and invalidates the previous V315/V316 numerical widths and claimed 27.1% technology-cap shrinkage. Their original files are retained for provenance, but must be treated as superseded and not cited as valid evidence. V319 reruns the analysis with origin-correct factors.

## Results under the corrected objective

All 372 province-month targets were feasible under all six information sets. The same 116 annual OD routes and 8-source latent variables were used throughout. Five public-report-like groups were thermal (coal, gas, bioenergy, other fossil), hydro, nuclear, wind, and solar. The two-copy LP reports the worst-case conditional diameter after fields are observed.

| Supply information | Extra monthly receipt fields | Mean width | Median width |
|---|---:|---:|---:|
| Total generation cap | 0 | 2.4364 | 1.4330 |
| Five coarse categories | 0 | 1.8363 | 1.0517 |
| Eight detailed sources | 0 | 1.8046 | 1.0197 |
| Total generation cap | 341 | 2.4354 | 1.4328 |
| Five coarse categories | 341 | 1.6634 | 0.9689 |
| Eight detailed sources | 341 | 1.1472 | 0.7414 |

Paired over the same 372 province-month cells, replacing the total-only cap with the five-category cap reduced the bound diameter by a mean **21.5%** (median 21.6%; 268/372 cells narrowed). A province-cluster bootstrap 95% interval for the mean paired reduction was **15.3%–27.9%**. Refining five categories to eight reduced the no-receipt diameter by only another **1.07%** on average (median 0.52%; bootstrap interval 0.65%–1.55%). This suggests the coarse grouping retains most of the *generation-composition* constraint within this LP.

Monthly receipt totals alone barely helped with total-generation caps (0.06% mean paired reduction, 0% median). With five-category supply caps already present, the same 341 receipt fields reduced the bound by **5.25%** on average (median 2.48%; 360/372 cells narrowed). The eight-source cap plus receipt totals reached a 22.1% mean reduction relative to the eight-source cap alone, showing the more detailed source margins make receipt totals more useful. These are conditional LP diameter changes, not forecast-interval coverage or measured CIF accuracy.

## What this supports

This corrects and strengthens the useful part of the prior result: a coarse five-category monthly schema still meaningfully restricts source allocation in the declared feasible set, and its incremental value is much larger than receipt totals when the latter are observed without composition constraints. Fine coal/gas/etc. splits add little to the *supply-cap-only* diameter here, but become more valuable jointly with measured destination receipts.

It does **not** show a new prediction algorithm or validate hourly CIF. The target denominator is local monthly generation, not consumption; the bound can exceed 1 kgCO₂e/kWh because it is incoming emissions divided by local generation, a direct-import exposure ratio. Ember's generation and emissions are estimates from the same product, generation is treated as available export capacity, and the annual OD/monthly supply scopes are not independently reconciled. The five-group map is only a schema proxy; the user's original table still needs a source/definition/unit/release-time audit. No hourly outcome or independent route data enters this experiment.

## Research decision

The measurable contribution candidate is now an **information-budgeted identifiability benchmark**: quantify the value and complementarity of coarse monthly category supply statistics and monthly destination receipts, with sharp conditional bounds. The LP itself is standard, so this alone does not support a novel forecasting-method claim. To support a stronger paper, the next gate must use an independent, same-boundary source for monthly generation/flows and a consumption denominator, then validate coverage and efficiency against held-out hourly labels (or explicitly position the paper as a data-identifiability benchmark if such labels are unavailable). The central claim must remain about bounded monthly direct-import exposure until those gates pass.

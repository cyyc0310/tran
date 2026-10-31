# V320 results: public monthly generation exists, but its scope is not an all-plant export cap

Run date: 2026-10-30. Protocol: [V320](V320_NBS_MONTHLY_GENERATION_SCOPE_AUDIT_PROTOCOL.md). Main output is in `results/v320-oreo-official-monthly-generation-audit-r2/`; the downloaded raw CSV snapshots are in `results/v320-oreo-official-monthly-generation-audit-r1/`.

## Source and monthly schema

The OREO monthly-generation table is NBS-derived, CC BY 4.0, and lists total, thermal, hydro, nuclear, wind, and solar generation for all 31 provinces. OREO's methodology says monthly generation covers industrial enterprises above designated size, while its annual generation table covers all power plants; the monthly series is therefore lower than the full annual system output. It also says the February generation value is cumulative January–February, so there is no distinct January row. We respected that by summing February once plus March–December; we did not manufacture separate January and February values. The NBS monthly production bulletin likewise defines the industrial reporting scope as enterprises above the designated-size threshold and notes annual coverage changes can affect comparisons. [OREO monthly dataset](https://oreo-data.org/dataset/cn-prov-monthly-generation-by-tech), [OREO methodology PDF](https://oreo-data.org/files/methodology/Methodology%20for%20China%20Datasets.pdf?v=5a79a21f), [NBS monthly bulletin](https://www.stats.gov.cn/sj/zxfb/202610/t20261019_1965613.html).

In 2023 the file has 11 reporting bins (February cumulative through December) for total, thermal, nuclear, wind, and solar in all 31 provinces. Hydro is present for 30 provinces; Tianjin is missing throughout. The five technology categories reconcile closely to the reported total, but small rounding differences make their sum exceed the total in 16 provinces (maximum absolute difference 0.082 TWh).

## Scope comparison and supply-cap implication

After treating February as January–February, monthly above-designated-size generation sums to **8,854.86 TWh**. The same provider's 2023 annual all-plant generation is **9,456.44 TWh**. Thus the monthly table covers **93.1% on average by province** (range 84.3%–99.0%); the aggregate shortfall is **601.58 TWh (6.36%)**. Province ranks align closely (Pearson `r=0.9991`), but high rank correlation does not repair the missing generation. Ember's 2023 monthly estimate is 4.82% above the OREO monthly-derived sum; V310's annual dataset sum is 6.33% above it.

Because monthly NBS data omit below-designated-size generation, it is **not a strict upper bound on all provincial generation or exports**. Using the observed thermal/hydro/nuclear/wind/solar values as a hard all-plant supply cap would falsely declare unobserved generation unavailable. V319's 21.5% five-category diameter reduction is still a structural finding for its Ember-based declared feasible set; it cannot be transferred numerically to this NBS-derived field contract without adding an unobserved-generation remainder.

## Can annual NBS totals repair the consumption-side boundary?

The same OREO release provides public NBS-derived annual all-plant generation and annual electricity consumption. For 2023 these sum to **9,456.44 TWh** and **9,223.80 TWh**, respectively. Combining those with the V310 annual OD matrix, the diagnostic `generation + OD net imports − consumption` is negative in **8 of 31 provinces** (negative residuals sum to −97.85 TWh; positive residuals sum to +330.49 TWh; mean absolute provincial residual 13.82 TWh). National totals being close is not enough to reconcile provincial flows. Because V310 OD has not been proven to share the NBS consumption/generation accounting boundary, this does not establish which source is wrong; it rejects direct, unqualified fusion of the records as a strict provincial consumption-CIF balance.

OREO's monthly consumption page is marked CC BY 4.0 but currently requires an access request. The public annual consumption table therefore provides a level check, not a month-by-month demand denominator. [OREO annual generation](https://oreo-data.org/dataset/cn-prov-annual-generation-by-tech), [OREO annual consumption](https://oreo-data.org/dataset/cn-prov-annual-consumption), [OREO monthly consumption page](https://oreo-data.org/dataset/cn-prov-monthly-consumption).

## Research consequence

This advances the data path: a reproducible, broadly available five-category monthly source exists. It also sharpens the missing-information question. To turn public monthly categories into credible source-risk bounds, one must represent (a) the monthly generation outside the reported above-size scope, (b) the unobserved timing and technology mix of that residual, and (c) the missing Tianjin hydro series, rather than silently treating these as zero. **V321 has now added the annual all-plant minus observed monthly-output remainder as an explicit unobserved supply pool**; it reduces category-field shrinkage from V319's 21.5% to 3.62% on average. The annual OD remains a separate, unverified accounting input; the outcome remains an import-exposure proxy, not standard CIF or hourly validation.

## Reproducible artifacts

- Script: `scripts/v320_nbs_monthly_generation_scope_audit.py`
- Protocol: [V320](V320_NBS_MONTHLY_GENERATION_SCOPE_AUDIT_PROTOCOL.md)
- Summary: `results/v320-oreo-official-monthly-generation-audit-r2/summary.json`
- Province table: `results/v320-oreo-official-monthly-generation-audit-r2/province_annual_scope_comparison.csv`
- Monthly coverage: `results/v320-oreo-official-monthly-generation-audit-r2/2023_monthly_coverage.csv`
- SHA-256 hashes for all five input datasets and the audit script are recorded in `summary.json`.

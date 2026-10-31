# V304：区域送受电边际总量能否给进口电平均因子有用的严格界限？

Frozen 2026-10-30 before solving. This is an information-identification experiment using 2023 regional aggregates; it is not a province-level or hourly CIF forecast.

## Question

The official provincial-factor formula includes local direct-generation emissions and imported electricity weighted by source provincial or regional emission factors. Public 2023 tables provide regional sending/receiving margins and a set of region-to-region routes, but not each route's transferred volume. Can the missing origin-destination matrix be treated as a transportation polytope, yielding informative sharp bounds on each receiving region's import-weighted mean CO₂ factor?

## Frozen data and constraints

- Source emission factors: the 2023 MEE/NBS official regional average CO₂ factors (seven regions).
- Supply and demand margins plus route support: Table 1 and Table 2 of Quan, Jin & Xu (2025), *Electric Power* 58(10):63–70, DOI 10.11930/j.issn.1004-9649.202504020. Volumes are in 10,000 kWh.
- Use the article's listed region-to-region route support as the primary feasible graph. Also run a dense off-diagonal graph using the same margins as a conservative sensitivity; this avoids relying solely on structural zeros inferred from the table's route listing.
- For each feasible nonnegative flow matrix, enforce exact origin outgoing totals and destination incoming totals. For each destination with positive inflow, minimize and maximize `sum(origin EF × flow) / destination inflow` with linear programming. Because the denominator is fixed by the receiving margin, these are sharp extrema over the declared feasible set.
- Verify both models are feasible and that total outgoing and incoming margins match exactly. Report all 5 receiving-region bounds and widths. Do not fit to destination target EFs.

## Decision interpretation

This experiment identifies only the carbon-factor range of imported electricity under incomplete flow attribution. It does not estimate the receiving region's full average factor, since local direct-generation emissions and provincial subregion exchanges are not reconstructed. Bounds are useful for the information-budget question if route constraints materially narrow them versus dense support and some are narrow relative to their source-factor range; wide bounds would quantify why the current marginals are insufficient.

## Primary sources

- MEE, 2021 electricity CO₂ factor calculation method, §I.1–3 and data sources: https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202404/W020240412827267566470.pdf
- MEE/NBS 2023 regional factor table: https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/W020251231726284332528.pdf
- Quan, P., Jin, Y., & Xu, S. (2025), DOI 10.11930/j.issn.1004-9649.202504020: https://www.electricpower.com.cn/article/doi/10.11930/j.issn.1004-9649.202504020

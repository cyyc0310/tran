# V303：2023省级受入电量能否解释官方EF迁移残差？

Frozen 2026-10-30 before analysis. This is a post-hoc diagnostic on the completed V300 2023 province-held-out predictions. It does not constitute a prospective feature test or a blind region test.

## Fixed question and data

Use V300's frozen 2023 predictions from the five-group lagged-generation-mix model. Join its 30 official MEE/NBS factor provinces to (a) 2023 provincial social electricity consumption in *China Electric Power Statistical Yearbook 2024* table 3-1, and (b) 2023 province-level electricity receipt volume reported in table 3 of Quan, Jin & Xu (2025), *Electric Power*, 58(10):63–70, DOI 10.11930/j.issn.1004-9649.202504020. The article reports that its regional/provincial exchange-volume data come from the *China Electric Power Statistical Yearbook* and Beijing Power Exchange Center sources.

The only added variable is `receipt_share_pct = 100 × province receipt_10k_kWh / (social consumption_100m_kWh × 10,000)`. It is same-year 2023 information and must be treated as unavailable to the lagged 2022-mix forecast. It is used solely to diagnose whether the known V300 errors align with direct receipt volume.

Pre-specified analyses:

1. Spearman association of receipt share with signed and absolute frozen V300 coarse-five residual.
2. One-feature, leave-one-province-out linear residual correction: for each held province, train an intercept and standardized receipt-share slope on the other 29 frozen V300 residuals, and add that correction to its frozen V300 prediction.
3. Show all provinces and the largest errors, with no subgroup tuning or additional predictors.

## Interpretation limit

Receipt volume is not the same as net imports, does not identify sending province or electricity source carbon intensity, and may reflect a specific transmission/market accounting scope. Any same-year association is explanatory/post-hoc only. Even a clear result would motivate testing source-resolved incoming energy in an operational budget; it would not validate hourly CIF or establish causal attribution.

## Source

- Quan, P., Jin, Y., & Xu, S. (2025). “区域碳双控下跨区电力间接碳排放核算方法.” *中国电力*, 58(10), 63–70. https://doi.org/10.11930/j.issn.1004-9649.202504020
- Table 3 is titled “跨省电力送受情况” and lists province-level sending and receiving volumes (10,000 kWh). The 2023 electricity-consumption denominator is transcribed from CEC, *China Electric Power Statistical Yearbook 2024*, table 3-1.

# V409: reconciliation of the Shanghai monthly NBS files supplied on 2026-10-31

The three supplied files are exports of current monthly values for thermal, wind, and solar generation. Their 33 non-empty observations (11 months × 3 technologies) exactly match the Shanghai records retrieved from the National Bureau of Statistics interface in V392. They add no new months to the research dataset. SHA-256 hashes of the supplied files:

| File | SHA-256 |
|---|---|
| `火电-分省月度数据.csv` | `92e02b10dbf5d19f3a680b481e005e5b6ada8a32c97d0bc7f805076cb5b10228` |
| `风电-分省月度数据.csv` | `7a6acef4e1503f7b80b7350adaaaea12a7bfe0fec639467c9532b99bdca11ad9` |
| `太阳能-分省月度数据.csv` | `85941ab60c7a99bdeab88d43ca3c386cc409e5fe3a018640900a4abf3fdd8407` |

The supplied period is September–December 2025 and March–September 2026. The export has no January–August 2025 rows; the current-value field is blank for February 2026. The files contain the NBS unit `100 million kWh`, current values, cumulative values, and growth rates. The NBS note scopes monthly energy-product production to above-designated-size industrial units; these values are not demonstrated to cover every generator in the regional electricity system.

For the 11 matching months, the official current-value panel reports hydro and nuclear as numeric zero, consistent with the user's clarification. The official total-generation series is also needed to form a denominator; it is present in the V392 official snapshot, not in these three attachments. Across these 11 months, the observed thermal/total share is 94.13%–96.86%. Under the V392 illustrative thermal EF support `[400, 1000] gCO₂/kWh` and zero operational EF for nonthermal technologies, the lower endpoints of the generation-weighted monthly mean support range from 376.5 to 387.4 gCO₂/kWh, and the upper endpoints range from 941.3 to 968.6 gCO₂/kWh. Every lower endpoint exceeds 300. This supports a coarse *monthly mean-support state* under those explicit assumptions, not a measured monthly EF or a statement about the hourly CIF distribution.

The data cannot distinguish coal from gas or oil within thermal generation. They also omit hourly generation, hourly thermal dispatch, imports/exports, storage behavior, and direct hourly emissions. Therefore they cannot supply the V408 threshold-side field (coal+oil at 600; coal at 900), nor can they certify hourly threshold counts or interval coverage. The complete API reconciliation and its longer time-series context are documented in [V392](V392_NBS_OFFICIAL_LONG_PANEL_AUDIT.md) and [V394](V394_ALL_PROVINCE_RELEASE_LAG_STATE_RESULTS.md).

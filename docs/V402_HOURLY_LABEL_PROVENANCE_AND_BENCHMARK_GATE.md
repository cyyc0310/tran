# V402: China hourly-label provenance and benchmark feasibility gate

Date: 2026-10-31. This is a source-provenance and task-definition audit. It does not claim a new prediction result.

## User-provided Shanghai exports

The current Desktop files `火电-分省月度数据.csv`, `太阳能-分省月度数据.csv`, and `风电-分省月度数据.csv` are byte-identical to the archived inputs under `data/nbs_user_exports/shanghai/`:

| File | SHA-256 | Archived file |
|---|---|---|
| 火电-分省月度数据.csv | `92e02b10dbf5d19f3a680b481e005e5b6ada8a32c97d0bc7f805076cb5b10228` | `thermal.csv` |
| 太阳能-分省月度数据.csv | `85941ab60c7a99bdeab88d43ca3c386cc409e5fe3a018640900a4abf3fdd8407` | `solar.csv` |
| 风电-分省月度数据.csv | `7a6acef4e1503f7b80b7350adaaaea12a7bfe0fec639467c9532b99bdca11ad9` | `wind.csv` |

The exports contain Shanghai current-month and year-to-date rows, in 100 million kWh, for 2025-09 through 2026-09 with gaps in the displayed period; February current values are blank because the NBS reporting bucket is January–February cumulative. They carry the above-scale industrial-entity coverage note. The files do not include province name, total generation, hydro, or nuclear columns; Shanghai attribution comes from the user's description, and hydro/nuclear zero values were separately checked in the NBS official snapshot (reported zero is not imputed into missing months). The three files add no new sample or field to V378/V391/V394.

The official NBS monthly endpoint was already queried for all 31 mainland provincial-level regions in V394 (2018-01 to 2026-09, retaining missing current-period values). Therefore this re-upload does not block the regional monthly-composition work. The main unresolved data limitation is a full-scope monthly generation series and an independent hourly target.

## What higher-resolution Chinese references actually provide

1. The 2025 *Strategic Study of CAE* article on time- and region-specific electricity carbon factors describes a physical accounting method using time-varying generation and interregional power-flow data. Its Jiangsu demonstration reports four seasonal typical days in 2023 and hourly city-level calculations for 13 cities. It explicitly characterizes these outputs as calculated factors from grid topology, power flows, generation, and source emission factors, rather than a province-wide released hourly observation series. This may become a disclosed mechanism-based reference set if the underlying inputs/outputs can be obtained, but it is not an independent observed hourly truth for a blind test. Source: [article and methods](https://www.engineering.org.cn/sscae/CN/1177188236516242258), especially sections 3 and 4.
2. The 2026 *Scientific Data* release provides 8,760 hourly CIF values for 31 provinces in five scenarios and years 2025–2060, generated through generation-transmission-load-storage planning, hourly dispatch simulation, and carbon-flow calculations. Its technical validation compares simulated outputs with annual official factors for 2021–2023; it does not validate hourly values against observed hourly CIF labels. It is useful as a model-generated stress-test/reference domain, but using it as ground truth would measure agreement with that simulator. Source: [paper](https://www.nature.com/articles/s41597-026-07272-6).
3. The accompanying Tsinghua data release includes provincial factors, technology-specific records, and interprovincial transfer matrices, but its landing page does not identify an hourly historical truth product. Source: [Zenodo dataset](https://zenodo.org/records/20344652).

These references narrow the novelty claim. “Produce high-frequency Chinese provincial CIF” is not a defensible standalone novelty claim: simulated hourly provincial factors are now available, and grid-flow accounting and time-specific factor calculations are established. A defensible benchmark contribution must distinguish labels by provenance and target boundary, and test whether a method trained under one information budget transfers to independent held-out targets without silently treating modeled values as observations.

## Updated research judgment

The Shanghai NBS exports strengthen the reproducibility of the low-information input, but they do not strengthen hourly prediction evidence. The strongest empirically supported contribution remains a release-time-aware, province-disjoint benchmark for what public monthly composition identifies about annual/monthly CIF support or threshold state. V394 provides the all-province monthly panel and the temporal release rule; V401 shows why production and delivered-consumption targets diverge, including the large Shanghai scope and import gaps. Neither establishes probabilistic hourly coverage or an algorithmic breakthrough.

The proposed benchmark should keep three target tiers separate:

| Tier | Target | Suitable evidence | Claim allowed |
|---|---|---|---|
| A | Monthly or annual production-side factor/support | Official NBS generation mix and official annual factors with matching boundary | Low-information regional estimation/identification, with release-time and scope constraints |
| B | Hourly mechanism-derived CIF | Published/grid-calculated hourly factors with full provenance, such as the Jiangsu seasonal-day study or a simulator release | Agreement/transfer to a disclosed reference model, not observed-truth accuracy |
| C | Hourly delivered/consumption-side CIF | Independent metered hourly generation, imports/flows, emissions or official delivered-factor series | Genuine hourly predictive coverage and sharpness for a clearly stated estimand |

Do not pool tiers B and C into a single score. For tiers A/B, report a strict province-held-out and future-time split, plus a separate scope mismatch audit; tier C is the gate for any claim of trustworthy hourly prediction. A simulator can support stress tests and sensitivity analyses, but not close tier C.

## Next concrete gate

1. Preserve the V394 all-province panel and V401 Shanghai boundary audit as the Chinese low-information benchmark track.
2. Search for a documented, permission-compatible Jiangsu hourly calculation release and its input vintages. If the underlying hourly outputs cannot be obtained, make no city-hour experiment from figures.
3. If obtained, freeze a separate mechanism-reference experiment with province/city and season holdouts, compare public monthly mix baselines, a standard temporal model, and a flow-aware reference. Label every score “reference-model agreement.”
4. Keep the actual hourly-coverage claim blocked on a tier-C target source; do not tune a model against the 2026 projected CIF dataset and present the result as validation.

## Reproducibility note

This gate was based on SHA-256 comparisons and the cited paper/dataset descriptions. It did not download or score any new hourly target, run another model, or change the production predictor. The V378 export parser and V394 all-province snapshot are the reproducible code/data path for the already-established findings.

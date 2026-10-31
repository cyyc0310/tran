# V99: Chinese monthly fuel groups can cross the risk threshold under factor uncertainty

## Question

V97–V98's candidate summary is monthly MWh on the fuel side whose emission-factor range lies above a target threshold. This only defines a stable field if the local factor support does not cross that threshold. V99 checks this requirement on the frozen Ember China province-month panel audited in V60. It is a data-contract diagnostic, not hourly prediction validation.

## Data and sensitivity assumptions

The frozen input contains 31 provinces, 2016-01 through 2026-09, and 4,128 complete province-months. Each month has eight atomic generation groups plus a total. Their sum reconciles to the total within the V60 rounding tolerance (maximum absolute difference 0.003 TWh). We exclude national aggregate rows.

The source distinction matters for a China paper: the National Bureau of Statistics' monthly release reports above-scale industrial generation by thermal, hydro, nuclear, wind, and solar categories, with an explicit above-scale-enterprise scope; it does not establish publicly reported monthly provincial coal/gas splits. Ember's China explorer does publish province-level generation/emissions and is updated monthly, but documents that the estimates combine public sources and estimation steps. Therefore the eight-class input used here is a secondary estimated panel, not equivalent to a directly measured official coal/gas province-month table. See the [NBS September 2026 release](https://www.stats.gov.cn/sj/zxfbhjd/202610/t20261019_1965613.html) and [Ember China data explorer/methodology](https://ember-energy.org/data/china-electricity-data-explorer/).

To make the threshold crossing explicit, we use the following deliberately broad sensitivity intervals around the project's operational proxy factors: coal [900, 1100], gas [360, 440], other fossil [200, 600], and zero for bioenergy, hydro, nuclear, wind, and solar. These are not official Chinese factors or a validated uncertainty distribution. A group is “definitely above” only if its lower endpoint is strictly above τ, “definitely below” if its upper endpoint is at or below τ, and otherwise “ambiguous.”

## Results

| τ (gCO2/kWh) | Definitely-above groups | Ambiguous group | Ambiguous share of all generation | Mean ambiguous share per province-month |
|---:|---|---|---:|---:|
| 300 | Coal, Gas | Other fossil | 0.82% | 0.79% |
| 400 | Coal | Gas, other fossil | 4.01% | 6.45% |
| 500 | Coal | Other fossil | 0.82% | 0.79% |
| 600 | Coal | None | 0.00% | 0.00% |
| 800 | Coal | None | 0.00% | 0.00% |
| 900 | None | Coal | 62.11% | 58.10% |
| 1000 | None | Coal | 62.11% | 58.10% |

At τ=600, the risk-side local statistic has a clean definition under this assumed support: monthly coal generation is the only definitely-above category. At τ=900, the ±10% coal-factor band touches the strict threshold; coal therefore becomes ambiguous, and it accounts for 62.1% of energy in the panel. The ambiguous annual energy share at τ=900 declines from 69.4% in 2016 to 55.1% in 2025 and is 53.4% in the partial 2026 period, remaining substantial throughout. This is a sharp sensitivity to factor support, not evidence that actual Chinese coal factors have this uncertainty.

## Interpretation for method design

The risk-side statistic has a precise conditional interpretation: it is a near-sufficient monthly compression only after the fuel-factor support and system boundary are specified. For thresholds near a dominant category's uncertain factor, factor uncertainty can overwhelm the compression advantage. A credible method should therefore expose (a) a local factor interval/source and vintage, (b) the generation-vs-consumption estimand, and (c) an explicit abstention or wider set when a category crosses τ. The public Ember panel cannot establish an as-of release history and its fuel splits are modeled; the user's original monthly table still needs its publisher, original header, units, release date, and revision policy audited.

Nothing here identifies hourly dispatch or CIF tails. There is no hourly label, no coverage score, and no prediction comparison in V99. The proposed contribution remains an information-budget/partial-identification benchmark candidate, not a validated algorithmic breakthrough.

## Reproducibility

- Protocol: `docs/V99_CHINA_FACTOR_THRESHOLD_STABILITY_PROTOCOL.md`
- Script: `scripts/v99_china_factor_threshold_stability.py`
- Frozen output: `results/v99-china-factor-threshold-stability-r1/`
- Protocol SHA-256: `4b9f8d8883d769407876da0c8ede96c30af5ba938ee3504d34bb3fc1afd5cbbf`
- Script SHA-256: `5f0c90c00dde988f56b8ee04cb77598a5f7a186c797e4e2f0b560a78bddbb45d`
- Summary SHA-256: `0ccbf42f5518cab3b5a532be3fea0597395e5bebc98ad0d0215ad154a84e0ff3`

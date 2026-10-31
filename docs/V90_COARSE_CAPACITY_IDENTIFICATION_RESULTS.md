# V90: annual coarse technology capacity retains useful but weaker tail identification

**Status: mechanism candidate, not a paper breakthrough or China validation.** Frozen 2026-10-28. Results use the previously opened 25 EIA balancing authorities, 2024–2025 hourly fixed-EF proxy labels, and one-year-lagged EIA-860 capacity. The tested coarse annual information is a shared capacity for thermal, nuclear, hydro, wind, and solar; thermal capacity is shared across coal, gas, oil, and other. This is an analogue of a coarse public technology table, not evidence that Chinese monthly-generation records and annual installed capacity have matching province boundaries or release dates.

## Main result

Among 589 region-months with at least 80% hourly coverage, 358 (60.8%) pass the retrospective coarse-group peak and monthly-energy feasibility audit. The audit compares lagged annual capacity against target-month hourly labels, so this 60.8% is not prospective availability. On these 358 months, the monthly-fuel-margin baseline is compared with the coarse-capacity outer set under identical target and sample definitions:

| Threshold | Tail target | Monthly margins positive / negative / abstain | Coarse capacity positive / negative / abstain | Mean K-set width, monthly → coarse |
|---:|---:|---:|---:|---:|
| 100 | ≥1% | 5.2 / 19.0 / 75.8% | 75.8 / 19.0 / 5.2% | 544 → 419 h |
| 100 | ≥5% | 5.2 / 19.0 / 75.8% | 75.2 / 19.0 / 5.8% | 544 → 419 h |
| 100 | ≥10% | 5.2 / 19.0 / 75.8% | 73.7 / 19.0 / 7.2% | 544 → 419 h |
| 300 | ≥1% | 5.2 / 19.0 / 75.8% | 51.8 / 19.0 / 29.2% | 544 → 442 h |
| 300 | ≥5% | 5.2 / 19.0 / 75.8% | 40.5 / 19.0 / 40.4% | 544 → 442 h |
| 300 | ≥10% | 5.2 / 19.0 / 75.8% | 30.7 / 19.0 / 50.2% | 544 → 442 h |

The average interval shrinks about 23% at τ=100 and 19% at τ=300. This is a clear information-value result: annual coarse capacity can constrain hourly tail counts beyond monthly fuel totals in this retrospective EIA proxy. It is not an improvement in a learned predictor, and it does not establish calibrated forecast coverage.

## Same-month fuel-specific comparison

The stricter fuel-capacity audit passes 326/589 months (55.3%). On the exact same 326 months, the monthly baseline has mean width 542 h; fuel-specific capacity has 362 h, while coarse-group capacity has 419 h. The coarser group therefore gives up about 57 h of width relative to fuel-specific capacity but keeps much of its value. At τ=100 and ≥1%, the coarse and fine positive-certification rates are both 75.8%; at τ=300 and ≥10%, they are 30.7% and 40.9%, respectively. Coarse capacity is eligible on 32 more retrospectively audited months.

Across all paired rows, coarse-set endpoints contain the fuel-specific endpoints (zero endpoint-containment failures); observed K is contained by both sets. The paired baseline, fine, and coarse frontier is saved in `paired_fuel_vs_coarse_frontier.csv`. This check is specifically against the interval endpoints; the report does not claim that the relaxed coarse calculation is a sharp capacity-constrained set.

## What is and is not established

- **Established for the proxy model:** the monthly fuel margins plus lagged annual coarse technology capacity produce a narrower conservative tail-count set than monthly margins alone on the audited EIA subset. The result survives removal of fuel-specific capacity detail.
- **Not established:** Chinese provincial capacity tables can be matched to monthly generation categories, have compatible gross/net and exchange boundaries, and were published before each scored month. Existing evidence also does not establish hourly CIF coverage or future-region prediction.
- **Not a new forecasting algorithm:** the gain comes from a physical upper bound and partial identification. The potential contribution is a data-budget benchmark with a transparent availability-versus-efficiency tradeoff, plus a proof of which coarse statistics shrink the identified set. Novelty requires positioning against optimization-based partial identification and proving the group-bound construction sharply or characterizing its conservatism.
- **Selection warning:** region-month eligibility uses observed hourly group peaks, so the reported coverage and risk frontiers are retrospective diagnostic results. A deployable study must report failures as failures without target-hour screening.

## Reproduction

- Protocol: [V90 protocol](V90_COARSE_CAPACITY_IDENTIFICATION_PROTOCOL.md)
- Implementation: [V90 script](../scripts/v90_coarse_capacity_identification.py)
- Outputs: [`results/v90-coarse-capacity-identification-r5`](../results/v90-coarse-capacity-identification-r5/)
- `python -m py_compile scripts/v90_coarse_capacity_identification.py` passed; the frozen script completed with observed-K containment and V66 nesting assertions intact. No general test suite was run.

## Next scientific gate

Before freezing any method, map actual public China monthly/annual table fields, release timing, geographic boundary, and capacity units; then evaluate whether the observed five-group information contract exists for the same province-months. If yes, use a label-hidden or independently collected hourly target set and count all capacity-audit failures. If not, this result supports a cross-system information-budget benchmark only, not a China-use claim. A stronger method novelty path is to derive a sharp grouped-capacity K set including positive generation in both event and complement hours, and compare its information frontier with the standard optimization relaxation under blind region/time holdout.

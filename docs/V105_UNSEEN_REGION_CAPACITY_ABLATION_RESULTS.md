# V105 results: paired capacity information ablation on unseen regions

## Design

On the same five V104 regions and 30 eligible Jan–Jun 2026 region-months, compare exact monthly fuel margins alone against monthly fuel margins plus lagged coarse technology-group capacity. Both conditions use the same nominal factors and the same coordinated ±10% thermal-factor stress box. The all-labels and K-set containment checks pass; the capacity set is nested inside the monthly-only set in all paired rows.

## Paired frontier

| Threshold | Monthly-only robust width | With capacity | Width reduction | Negative certificates, monthly → capacity | Robust abstention with capacity |
|---|---:|---:|---:|---:|---:|
| 600 gCO₂/kWh | 722.2 h | 434.1 h | 288.1 h (39.9%) | 0% → 0% | 100% |
| 900 gCO₂/kWh | 577.6 h | 270.0 h | 307.6 h (53.3%) | 20.0% → 20.0% at 1%/5%; 20.0% → 23.3% at 10% | 80.0% at 1%/5%; 76.7% at 10% |

There are no positive certificates. The capacity information sharply reduces the width of the partial-identification set, but does not create much new selective decision coverage in these regions. The small 3.3 percentage-point gain at τ=900 and a 10% tail threshold is concentrated in a narrow corner of the frontier. Macro figures weight each region equally; all five regions contribute six months.

## Interpretation

This separates the source of the V104 interval gain: the reduction is attributable to lagged coarse capacity constraints, not to a wider interval that merely covers labels. It still does not establish a useful predictor. Robust K containment remains a logical proxy-label check, not probabilistic calibration; the factor box is an uncalibrated stress assumption; and monthly target totals make this retrospective reconstruction. These five new regions support transfer of the set-narrowing mechanism but challenge any claim of broad actionable tail-risk certification.

For a paper, this is evidence for a partial-identification benchmark or decision-frontier contribution if paired with a clean theorem, realistic Chinese input contracts, and real hourly CIF outcomes. It is not enough to claim a breakthrough CIF prediction algorithm. A further algorithmic effort should target the abstention bottleneck with a minimum-information characterization, while preserving coverage guarantees; if no legally available local or annual field moves the frontier, that is an important negative result and a hard limit on the forecasting claim.

## Reproducibility

- Frozen protocol: `docs/V105_UNSEEN_REGION_CAPACITY_ABLATION_PROTOCOL.md`
- Runner: `scripts/v105_unseen_region_capacity_ablation.py`
- Outputs: `results/v105-unseen-region-capacity-ablation-r1/`
- Hashes and source audits are recorded in `summary.json`.

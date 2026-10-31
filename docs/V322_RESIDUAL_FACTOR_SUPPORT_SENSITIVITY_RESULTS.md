# V322 results: category information is positive across support scenarios, but highly heterogeneous

Run date: 2026-10-30. Frozen protocol: [V322](V322_RESIDUAL_FACTOR_SUPPORT_SENSITIVITY_PROTOCOL.md). V321's `[0,1.0]` results are reused unchanged; only the `[0,0.5]` and `[0,1.5]` LPs were rerun. Cellwise monotonicity audit passed: widening the residual EF support did not narrow any interval in any information arm. Full scenario summaries and per-cell files are in `results/v322-residual-factor-sensitivity-r1/`.

## Five-category cap value across residual EF supports

The percentages below compare five-category monthly caps with total-only monthly caps on the same feasible set, once without and once with 310 hypothetical recipient-period totals.

| Residual EF support (kgCO₂e/kWh) | Category gain, no receipts: mean / median | Category gain, with receipts: mean / median | 95% province bootstrap, no receipts | 95% province bootstrap, with receipts |
|---|---:|---:|---:|---:|
| [0, 0.5] | 10.75% / 10.33% | 15.99% / 17.28% | 7.57%–14.19% | 11.58%–20.47% |
| [0, 1.0] | 3.62% / 0.00% | 11.20% / 0.30% | 1.81%–5.79% | 6.44%–16.37% |
| [0, 1.5] | 2.79% / 0.00% | 8.78% / 0.21% | 1.41%–4.44% | 5.00%–13.02% |

All province-bootstrap intervals remain above zero, but the median no-receipt benefit falls to zero at supports `[0,1.0]` and `[0,1.5]`. The mean benefit also declines as the unobserved pool can carry more carbon. With receipt fields, the mean category advantage remains larger, but most median gains are small at the wider supports. Receipt totals alone still do not help the total-only arm in a material way.

## Research implication

The qualitative result is not an artifact of one exact upper endpoint: category constraints have a positive average effect across all three declared scenarios, and adding recipient totals strengthens that effect. The **size and prevalence** of the gain are support-sensitive, however. A defensible paper should report an information-support frontier, not one headline shrinkage percentage. The 0–1.5 interval is only a robustness scenario, not a validated engineering range.

## Novelty boundary from related work

Chinese consumption-side carbon-flow accounting is already an active field: a 2026 *Energy Policy* study builds an electricity-extended input-output model with interconnected grids and accounts for generation, transmission, distribution, and upstream lifecycle emissions across 30 provinces in 2018–2022; its data are available on request. The present candidate should therefore not claim a new carbon-flow accounting method. Its potentially differentiating question is what can be identified and credibly bounded from public, incomplete monthly fields when full-consumption/flow data are unavailable. Robust conformal methods for perturbed scores and partial-label training are also established; adding a generic conformal layer would not by itself establish novelty. [Energy Policy study](https://doi.org/10.1016/j.enpol.2025.114907), [robust conformal prediction sets](https://proceedings.mlr.press/v235/h-zargarbashi24a.html), [conformal prediction with partially labeled data](https://proceedings.mlr.press/v204/javanmardi23a.html).

This still does not produce a forecast method or hourly CIF validation. The open breakthrough is to connect this scope-aware identification frontier to an independently reconciled consumption denominator and genuinely observed hourly labels, or to define the paper explicitly as a public-data identifiability benchmark. The actual user-provided table's source and schema also remain unverified in the workspace.

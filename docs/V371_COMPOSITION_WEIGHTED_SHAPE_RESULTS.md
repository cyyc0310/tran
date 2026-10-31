# V371 results: composition-weighted donor shapes add negligible value

Protocol: [V371](V371_COMPOSITION_WEIGHTED_SHAPE_PROTOCOL.md). Runner: `scripts/v371_composition_weighted_shape.py`. Results: `results/v371-composition-weighted-shape-r1/`.

## Results

All shape variants use the same local `700 × thermal_share` monthly level; only the donor hourly-shape weighting changes. The equal-weight donor baseline has mean absolute tail-count errors of 383.8 hours at τ=300 and 34.3 hours at τ=500.

| Donor shape | τ=300 tail-count error gain | 95% region-cluster interval | Regions improved | Hourly MAE gain |
|---|---:|---:|---:|---:|
| Nearest 3 mix donors | 3.9 h | 0.5–7.7 | 5/7 | −0.4 g/kWh |
| Kernel, bandwidth 0.15 | 3.3 h | −0.3–9.3 | 4/7 | −0.4 g/kWh |
| Kernel, bandwidth 0.30 | 2.3 h | 0.7–4.1 | 4/7 | +0.1 g/kWh |

These reductions are about 0.6–1.0% of the equal-weight tail-count error and do not improve hourly CIF MAE. At τ=500 the tail-count gains were below one hour on average and all region-cluster intervals included zero; τ=700 had no events. The stopping rule is not met.

## Decision

Close the composition-weighted donor-shape route for this dataset. The robust-sized increment remains the local thermal-share level versus national/pooled share; weighting hourly donors by hydro/nuclear/wind/solar/other composition is too small to support a methods contribution. More elaborate bandwidth search would risk optimizing noise with only seven regions.

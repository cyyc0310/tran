# V135 — Coherent Factor-Support Bounds Across Time Blocks

## Question

Can a three-day risk-side set be tightened by enforcing that the unknown fuel emission factors are shared across every block in a region-month, instead of independently relaxing them block by block?

## Frozen design

- Use V129's fixed 358 region-month cohort, V132's three-day bins, τ=600/900, and the same EF intervals.
- Compare (a) V132's blockwise robust endpoint hull (the conservative reference), (b) a coherent all-low EF scenario applied to every bin and summed, and (c) a coherent all-high EF scenario applied to every bin and summed. The final coherent robust hull is the union envelope of (b) and (c).
- For the risk-side arms, high-side fuel mix remains hidden within each block. At each global EF endpoint, use the min/max EF among fuels on the high and low sides to conservatively represent all hidden within-side mixes. At τ=900, the all-low coal factor equals τ, so its strict-exceedance set is exactly K=0; the all-high endpoint provides the upper robust scenario.
- For the detailed-fuel oracle, hold the same all-low or all-high EF vector fixed across every block and use the exact fixed-factor no-cap K set per block before summing.
- Assert coherent risk-side bounds contain the observed nominal K and coherent detailed robust interval, and remain inside V132's independently relaxed blockwise risk-side interval. Preserve all frozen rows.

## Interpretation boundary

This is a robust-set refinement under shared factor uncertainty, not an empirical estimate of Chinese EF uncertainty. Data remain retrospective EIA oracle coarsenings, and no Chinese hourly CIF label or reporting-error validation is present.

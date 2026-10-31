# V108: sharp hourly-tail set method and paper gate

## Target quantity

For each hour `h`, define generation-based average intensity

`I_h = (sum_f e_f x_{h,f}) / (sum_f x_{h,f})`,

where `x_{h,f} >= 0` is generation by fuel/technology and `e_f` is a direct generation emissions factor. For threshold `τ`, the monthly tail count is

`K_τ = sum_h 1{sum_f (e_f - τ) x_{h,f} > 0}`,

with zero-generation hours explicitly assigned `I_h = 0` or omitted under a declared convention. This is an **average generation-based** intensity, not marginal emissions, consumption-based intensity, or lifecycle intensity.

## Exact identified set under observed monthly fuel margins

Inputs: `T` hourly slots; monthly fuel margins `M_f = sum_h x_{h,f}`; technology group `g(f)`; common group capacity upper bounds `C_g`; and fixed factors `e_f`. No minimum output, ramps, unit commitment, imports, or transmission constraints are assumed.

For a proposed event count `k`, aggregate event-block fuel allocations `z_f = sum_{h in E} x_{h,f}` must obey

`max(0, M_f - (T-k) C_{g(f)}) <= z_f <= min(M_f, k C_{g(f)})`,

and group-hour capacity must apply to each hour, not to total fuel independently. With fuel-specific monthly margins and a shared group capacity, the event and complement blocks can be checked with a linear allocation problem. Strict event/non-event inequalities and positive generation on both blocks must be preserved. The exact set is

`S_τ(M,C,e) = {k in {0,...,T}: there exists a feasible hourly allocation with K_τ=k}`.

The solver must return the complete integer set, not only endpoints. V102 matched V90's full integer set on 3,078 previously opened rows, but the proof and implementation audit must be presented as a theorem and standalone derivation in a paper appendix.

## Coarse thermal margins

If only thermal total `M_T` and clean technology margins are observed, and thermal factors are allowed anywhere in `[e_L,e_U]`, then the event/complement block problem reduces to thermal event energy `x` and clean event energy `y`. Capacity gives an axis-aligned feasible rectangle, and the complement non-event requirement clips it by one half-plane. The vertices of this clipped polygon suffice to decide whether each integer `k` is feasible. V106 implemented this construction for `[400,1000]` proxy gCO₂/kWh and found all observed proxy K values contained; this is an outer-contract theorem only if the factor-support and group-capacity definitions correspond to the stated physical system.

## Factor uncertainty

If `e` lies in a coordinatewise box `[e^L,e^U]`, the event condition is monotone in each factor for every fixed allocation. Therefore

`min S_τ(M,C,e^L) <= K_τ(e,x) <= max S_τ(M,C,e^U)`

for every admissible `(e,x)`. The endpoint interval is the sharp endpoint hull over the box. It is **not** generally proven that every integer between the endpoints is jointly attainable under one globally fixed factor vector. Positive/negative certificates based on the hull are valid only relative to the input contracts and assumed box.

## What the current evidence supports

- The algebraic set-shrinking mechanism is reproduced on five held-out EIA regions and 30 months. V105 shows the lagged group-capacity condition reduced the ±10%-factor-box width by 40%/53% at `τ=600/900` relative to fuel margins alone.
- Under coarse monthly thermal/clean margins and broad thermal EF support, capacity still shrinks intervals, but the τ=900, 10%-tail negative certificate rate is only 6.7%; nearly all other decisions abstain (V106).
- V104–V106 use fixed-factor EIA generation-based proxy labels, target-month monthly totals, and U.S. lagged capacity records. These do not prove real-China CIF accuracy, probability coverage, prospective forecasting, or transfer to marginal/consumption-based CIF.

## Publication gate and candidate claim

The plausible paper is an **information-sufficiency and partial-identification benchmark for hourly CIF tail risk under low-frequency fuel statistics**, with an exact set solver, factor-support uncertainty, capacity and fuel-resolution ablations, missing-input abstention accounting, and preregistered spatial/time holdouts. Novelty is not established yet: aggregate-data partial identification, temporal disaggregation, monthly grid factors, and hourly CIF forecasting all have prior work. The contribution needs a targeted systematic literature review and a direct comparison against linear/LP bounds plus standard low-frequency-to-hourly reconstructions.

Do not claim an hourly CIF forecasting breakthrough until all are met:

1. independently reproduce the theorem and complete-set solver on synthetic adversarial examples and a second implementation;
2. freeze a genuinely future-time holdout in addition to region holdouts;
3. evaluate against actual hourly **average generation-based** CIF labels in China or an explicitly scoped, licensed validation panel;
4. report coverage-width/selective-risk tradeoffs against monthly-constant, donor-shape, temporal-disaggregation, quantile, and conformal baselines under the same information budget;
5. show an information field that materially reduces abstention without using target-hour labels, and establish that its Chinese source, unit, vintage, and publication date are reproducible.

Until then, the current result is a useful mechanism candidate and a map of non-identifiability, not a validated forecasting breakthrough.

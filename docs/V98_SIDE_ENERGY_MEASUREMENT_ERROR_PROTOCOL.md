# V98 protocol: measurement uncertainty in threshold-side monthly energy

Frozen 2026-10-28. Stress relative error ±0/1/2/5/10% in the threshold-side monthly MWh while treating the broad thermal total and coarse technology capacities as exact. Use the V90 358-month cohort, V97's same-coarse-capacity detailed-margin comparator, τ=600/900, and 1/5/10% tail targets.

For a measured high-side total `Hhat`, the latent true amount lies in `[Hlo,Hhi]=[Hhat(1−ε),Hhat(1+ε)]` clipped to `[0,Mthermal]`. Do not select only one endpoint. For a candidate interior K, let `z` be event high-side generation. Any feasible monthly high-side amount can be reduced to the minimal `H=max(Hlo,z)`: this increases the amount of low-side generation available in the complement and weakly relaxes its non-event condition. The union is therefore the union of two cases:

1. `z<=Hlo`: use fixed monthly high-side total Hlo and the V96 exact threshold-side solver.
2. `z>=Hlo`: choose `H=z`, so all high-side monthly output falls in the event block; the complement then contains only below-threshold thermal and clean generation. Reduce this case to a one-dimensional feasible event-thermal interval and maximize its piecewise-linear event excess.

For K=0 use Hlo (least emissions); for K=T use Hhi (most emissions). Require the error-robust set to contain observed K, contain the zero-error V96 side set, and remain nested in the coarsest thermal-total-plus-capacity set. Zero error must exactly reproduce V96. Report all failures and compare to V97 detailed monthly margins plus the same coarse technology capacities on the exact same rows.

This assumes the total thermal generation is error-free and only the threshold-side aggregate is uncertain. It is a sensitivity envelope, not a measured China reporting error model. The EIA oracle/cohort/proxy limitations remain.

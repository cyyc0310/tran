# V348: sensitivity frontier for monthly Ember fuel-share error

Registered 2026-10-30 after V347 found a large conditional value for Ember's estimated fuel mix. The concern is whether that value survives misspecification of estimated monthly fuel shares.

For the same V341 330 province-periods and τ=300 g/kWh, hold Ember's monthly total thermal generation fixed and allow each of the four thermal fuel shares (coal, gas, other fossil, bioenergy) to vary within ±0%, ±10%, ±25%, ±50%, or ±100% of its Ember value, intersected with the probability simplex. Compute the minimum effective thermal factor using V99 lower EF endpoints and the maximum using upper endpoints via two small linear programs. The V342 sharp fixed-factor solver then gives a model-conditional K hull. Compare against V346's NBS-plus-residual hull under monthly-only and 2022-lagged-capacity arms.

Report feasible cell counts, factor intervals, widths and width reductions. Do not label these perturbation bands as empirical source-error intervals; no independent monthly fuel truth or hourly CIF labels are used. Preserve capacity-infeasible Ember cells as unavailable and report their count.

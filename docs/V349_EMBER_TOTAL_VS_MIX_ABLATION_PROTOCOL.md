# V349: separate complete monthly-total value from estimated fuel-mix value

Registered after V347/V348 suggested that Ember's estimated fuel panel plus capacity could materially narrow K hulls, while its total generation is about 5.8% above the NBS above-size monthly total. To identify which information component drives the apparent gain, compare on the same 328 province-periods feasible under V348's 2022 capacity arm:

1. V346 NBS five-category margins plus annual scope residual and residual EF support [0,1000].
2. Ember's complete estimated monthly total collapsed to five technology groups, with thermal EF support [0,1100] and zero direct operational EF for hydro/nuclear/wind/solar. This matches the outer envelope of the V348 fuel-resolved EF scenarios, so any width reduction in arm 3 is not caused simply by giving the mix arm a wider EF support.
3. Ember estimated fuel mix with ±100% share perturbations, as in V348.

Use τ=300 g/kWh, 2022 CEC group capacities, and the exact V342 fixed-factor K-set solver. Compare widths on identical province-period keys. The difference between (1) and (2) is a combined complete-total/scope and monthly-margin change; the difference between (2) and (3) measures the additional modeled fuel-share information under the same Ember total, factor-support envelope and capacity vintage. The fuel-resolved EF interval remains province-period-specific through the V348 ±100% share perturbation.

The first implementation used [400,1000] in arm 2 while the fuel-level scenarios spanned [0,1100]. That was not a matched-support comparison and is superseded by the r2 run with [0,1100]. Preserve r1 only as a diagnostic artifact; do not use it to attribute gains to fuel mix.

Do not interpret this as truth validation: Ember is an estimated secondary panel, its monthly total has a different coverage boundary from NBS, and the EF supports are scenarios. No Chinese hourly labels enter.

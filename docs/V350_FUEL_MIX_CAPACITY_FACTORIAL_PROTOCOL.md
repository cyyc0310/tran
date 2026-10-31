# V350 protocol: factorial information-value test for fuel mix × capacity

Registered after V349-r2. Question: does Ember-estimated fuel composition narrow the high-threshold K identification interval by itself, or mainly when combined with installed-capacity bounds?

## Frozen design

- Cohort: same 328 province-period keys feasible in V348's ±100% share-error lagged-2022-capacity arm and V349-r2.
- Target: τ=300 g/kWh.
- Generation margins: Ember 2023 monthly eight-source estimates collapsed to the same five technology groups, with Jan-Feb combined to match the NBS bin.
- Factorial 2×2: (A) collapsed thermal group with the common outer EF support [0,1100] vs (B) fuel-resolved effective-EF support under V348's ±100% share-error simplex; crossed with (1) no active capacity cap vs (2) lagged 2022 CEC group capacities.
- Identification: exact V342 K-set endpoints; compare widths key-by-key. The cross-difference measures whether fuel-mix information becomes more valuable under capacity constraints.
- Fuel-resolved cells are taken from the frozen V348 frontier; the collapsed group widths under both capacity arms are computed with the same solver, margins, group capacity maps, and common EF support.

This is a conditional information-value experiment, not a forecast-validation experiment. Ember is modeled information, the ±100% share band is analyst-defined, and the lagged-capacity arm is retrospective. No hourly China CIF labels or interval coverage are available in this design.

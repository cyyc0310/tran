# V102: exact capacity-constrained K sets match V100 on the un-screened cohort

## Question

V100's monthly fuel margins plus lagged coarse technology capacities improved the K-set frontier after removing target-hour peak screening. V102 asks whether that result comes from a conservative relaxation or whether the calculated K sets are exact under the stated model. It pairs the V92 exact solver with every V100 capacity-feasible row.

## Result

The comparison covers **513 region-months**, **22 regions**, and **3,078 region-month × threshold × tail-target rows**. It retains the 155 region-months that fail the target-hour peak-capacity audit. With the exact same monthly fuel margins, lagged coarse group caps, and fixed EFs:

- Exact and V100 lower/upper endpoints match on **100%** of rows.
- The complete integer K sets match on **100%** of rows, not just their endpoints.
- The actual hourly K is in every exact set (**100%** empirical containment).
- The exact set is nested in the monthly-margin set on every row.

Region-equal mean exact widths remain **423.48 h** at τ=100 and **439.73 h** at τ=300, exactly matching V100 and the reported un-screened capacity frontier. The earlier τ=100/300 certification rates therefore also remain unchanged: conditional on capacity-feasible inputs, the ≥1% positive rates are 75.9%/48.7%, with zero observed certificate errors.

## What this establishes

Under fixed EFs, exact monthly fuel totals, one shared hourly cap per technology group, nonnegative hourly generation, no cross-hour constraints, and positive generation in both event and complement hours, the V100 set is an **exact sharp identified set** for the number of hours above τ. The result no longer depends on the 358-month hourly-peak-screened sample: exactness holds across all 513 monthly-energy-feasible months, including peak-screen failures.

This is the strongest method result so far. It converts the candidate from a merely conservative capacity bound into a closed-form, assumption-transparent partial-identification result for hourly tail risk. It is not point prediction and does not show that an hour-level value is recoverable from monthly information.

## Novelty assessment

Optimization-based sharp identification from aggregate data and polyhedral restrictions is an established general approach; for example, recent aggregate-data work obtains bounds by optimization under shape restrictions ([Moon, 2026](https://doi.org/10.1080/07474938.2025.2604682)). Temporal disaggregation from low-frequency totals to high-frequency paths also has a mature literature ([Sax & Steiner, 2013](https://doi.org/10.32614/RJ-2013-028)). The potential contribution is therefore not “using optimization” or “monthly data cannot determine a unique hourly path.” The narrower candidate is an exact characterization of a **threshold-exceedance count** identified set from fuel-specific monthly margins and shared coarse capacity envelopes, plus an empirical information-budget frontier and input-availability/abstention accounting.

This novelty claim still needs a targeted literature search for threshold-risk bounds in power systems, and it needs tests with factor-support uncertainty and a truly independent region/time set. All current hourly labels are previously opened fixed-EF EIA generation proxies. No China or real-CIF validation is established.

## Reproducibility

- Protocol: `docs/V102_FULL_COHORT_SHARP_CAPACITY_PROTOCOL.md`
- Script: `scripts/v102_full_cohort_sharp_capacity.py`
- Frozen output: `results/v102-full-cohort-sharp-capacity-r1/`
- Protocol SHA-256: `d0806546de23725189886635e5bb770e6bab90ddf89d4dcf844901fe300ab81e`
- Script SHA-256: `6bbaf01ba798ae076b224131c8385b9ee21eb0d2db62a13854ca9e173a1907d3`
- Summary SHA-256: `363148a235cbe5803bc24f9e240e3267dab558326fd62173e37b7143626f47f5`

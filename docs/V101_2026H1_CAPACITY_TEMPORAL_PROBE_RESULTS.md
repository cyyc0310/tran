# V101: frozen coarse-capacity mechanism on 2026H1

## Design and contamination status

V101 applies the V100 rule unchanged to 2026-01 through 2026-06, using 2025 EIA-860 coarse technology capacity. No target-hour peak filter is used. The information contract is eight monthly fuel margins plus lagged thermal/nuclear/hydro/wind/solar capacity; the label is the same fixed-EF EIA generation proxy.

This is a **method-specific temporal transport probe, not a blind confirmation**. The 2026H1 hourly targets and regions were previously used in other prediction experiments in the repository, so the dataset is not globally untouched. No V101 capacity outcome was used to tune the frozen method before this run.

## Candidate accounting

The fixed 25-region roster yields 148 coverage-qualified region-months. Monthly group energy fits the 2025 capacity envelope in 124/148 (83.8%); 24 months are capacity-incompatible and count as unavailable/abstaining. Four regions (CHPD, FMPP, GCPD, GRID) have all six months incompatible. Only 77/148 months pass the target-hour peak-capacity screen, but that screen is used for description only and does not filter the scored sample. The 124 feasible months span 21 regions.

## Results

Conditional on the 124 capacity-feasible months, the hourly K set covers the observed K in **100%** of rows, with zero positive/negative certificate errors. Region-equal results:

| Threshold | Method | Mean set width (h) | ≥1% positive | Negative | Abstention | K-set coverage | Certificate errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | Monthly margins | 563.7 | 6.3% | 14.3% | 79.4% | 100% | 0 |
| 100 | Lagged coarse capacity | 445.6 | 77.8% | 14.3% | 7.9% | 100% | 0 |
| 300 | Monthly margins | 563.7 | 6.3% | 14.3% | 79.4% | 100% | 0 |
| 300 | Lagged coarse capacity | 440.5 | 41.9% | 14.3% | 43.8% | 100% | 0 |

When the 24 capacity-incompatible months are retained as abstentions and the full 148-candidate denominator is scored equally by region, the ≥1% positive rate is 65.3% at τ=100 and 35.2% at τ=300. At τ=100 the corresponding negative/abstention rates are 12.0%/22.7%; at τ=300 they are 12.0%/52.8%. No capacity certificate is emitted for an incompatible input.

## Interpretation

The capacity partial-identification mechanism has encouraging time-transport evidence: it narrows the K set and retains empirical coverage after moving from 2024–2025 capacity vintages to 2026H1 with a lagged 2025 inventory, without filtering on hourly peaks. The 16.2% capacity incompatibility rate and four entirely unavailable regions prevent presenting the mechanism as uniformly deployable. The 2026H1 cohort is reused and the target is an hourly proxy, so this result cannot support an unseen-region claim, true-CIF calibration, or a prospective forecast claim. Inputs contain the target-month totals and therefore correspond to post-period reconstruction/nowcasting.

## Reproducibility

- Protocol: `docs/V101_2026H1_CAPACITY_TEMPORAL_PROBE_PROTOCOL.md`
- Script: `scripts/v101_2026h1_capacity_temporal_probe.py`
- Frozen output: `results/v101-2026h1-capacity-temporal-probe-r1/`
- Protocol SHA-256: `5b6fbae3f3055142793062432a3966b80637069c2bb4c4ace08d28493571ec4c`
- Script SHA-256: `d459f9ece3c28ac8072382c465e92a41c8cfd32eb13e1f239a509a251fa36d76`
- Summary SHA-256: `51e01fe94af338b6e6a898af4dbd101d3ce71aa7a9a93ed3f65986b8229b0a65`

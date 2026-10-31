# V122 — Frozen Rule on October 2026 Month-to-Date Results

> V139 correction notice: source support calibration included capacity-infeasible months. A saved audit now recalibrates the cutoff on input-feasible source months; one PACE τ=900 row changes from [0,508] to [0,68], while coverage of saved target proxy endpoints is retained. This is a posthoc correction, not a new blind confirmation or full raw-data rerun. See [correction scope](CAPACITY_CONTRACT_CORRECTION.md).

## Data and denominator

The frozen cohort contained the 43 V52 regions with 2026H1 complete-proxy coverage ≥90%, plus NWMT. All 44 regions remain in the availability file. October-to-date valid proxy coverage was ≥80% for 40 regions; 37 of those also had feasible 2025 lagged capacity constraints. Four regions were coverage-ineligible; three more had unavailable/infeasible capacity constraints. The latest timestamp present in the EIA file is 2026-10-28 06:00 UTC. This is an incomplete month, with 624 valid hours for most scored regions.

## Frozen high-threshold rule

The method had been frozen before inspecting October K outcomes: capacity-only at τ=600; at τ=900 use anchor∩capacity unless the monthly support gap exceeds the q=20% leave-region-out boundary, then use capacity. Each target region was excluded from its source calibration radius and support boundary.

| τ | Method | Endpoint coverage | Mean interval width | 10% tail positive / negative / abstain |
|---|---|---:|---:|---:|
| 600 | Capacity-only frozen rule | 37/37 | 253.5 h | 2.7% / 43.2% / 54.1% |
|  | Plain anchor∩capacity | 37/37 | **195.9 h** | 5.4% / 43.2% / 51.4% |
| 900 | Capacity-only | 37/37 | 183.2 h | 0% / 48.6% / 51.4% |
|  | Plain anchor∩capacity | 36/37 | 46.8 h | 0% / 54.1% / 45.9% |
|  | **Frozen support-gap rule** | **37/37** | **89.8 h** | 0% / 48.6% / 51.4% |

At τ=900 the frozen method restores the single missed endpoint from SC by falling back to its physical capacity set. It widens relative to plain intersection but remains 51.0% narrower than capacity-only. Three rows triggered fallback: JEA and SC, which had nonzero factor-box upper support, and PACE, whose proxy K was zero. The three region-months with nonzero factor-box upper support were JEA, LGEE, and SC; all three were contained. LGEE's anchor interval already spanned its capacity range, so the method retained that interval. The proxy factor-box upper endpoints were 3 h for JEA, 502 h for LGEE, and 162 h for SC.

At τ=600 the frozen rule intentionally uses capacity-only because V120 showed little reliable gain from anchoring at this threshold. It maintains 37/37 coverage but does not realize the narrower 195.9 h plain-intersection interval. Across the three decision rates (1%, 5%, 10%), the τ=900 method had no positive certificates; negative certificates were 48.6% and abstentions 51.4%.

## Interpretation

This is the strongest temporal confirmation so far for the risk-support gate: after freezing it, a new month-to-date window contains a nonzero τ=900 risk-support case, and the gate repairs the plain intersection's observed undercoverage while preserving about half the width reduction against the physical set. It is not a full-month or month-ahead forecast, and it uses U.S. EIA fixed-factor proxy labels and capacities. The region cohort overlaps regions scored in Q3; therefore this confirms time transfer, not transfer to unseen BA identities. Only three of 37 capacity-feasible rows had nonzero τ=900 factor-box upper support, so the risk coverage sample remains small.

The current candidate breakthrough is a threshold-specific support-aware partial-identification rule: use the narrow calibrated anchor only while its gap to the physical feasible upper support remains inside the source-region support boundary; otherwise return to the exact physical set. The next validation must be a complete untouched quarter and a spatial holdout not used in development, with enough nonzero high-threshold months to test that the gate detects support failures reliably.

## Reproduction

- Frozen protocol: `V122_OCTOBER_MTD_FROZEN_RULE_CONFIRMATION_PROTOCOL.md`.
- Runner: `../scripts/v122_october_mtd_frozen_rule_confirmation.py`.
- Row-level scores, full 44-region availability denominator, summary and input hashes: `../results/v122-october-mtd-frozen-rule-confirmation-r1/`.

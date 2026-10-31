# V121 — Frozen Support-Gap Rule on NWMT 2026Q3

## Freeze and scope

Freeze the V120 candidate before scoring NWMT 2026Q3 hourly K values:

- τ=600: use the exact factor-box capacity-constrained set directly; V120 did not show enough gain from anchoring at this threshold.
- τ=900: use the anchor-capacity intersection unless the support-gap feature exceeds the 20% source-region support boundary; above the boundary, fall back to the exact capacity set.
- For the target NWMT, exclude NWMT entirely from the source calibration radius and support-boundary pool. With 24 remaining regions, the anchor radius uses α=10%, rank 23; the support boundary uses q=20%, rank 20.
- Keep the V120 ±10% thermal EF box, lagged 2025 capacity vintage, V92 sharp set, and ≥80% complete-proxy month eligibility.

The selection roster was fixed using only completeness metadata: NWMT had <90% complete-proxy coverage in 2026H1 but 100% valid proxy hours in each Q3 month, and was not one of V118's scored targets. No Q3 intensity or K outcomes were used in selection.

## Information and evaluation

Input: NWMT monthly fuel margins, 2025 lagged group capacities, and source-region calibration records through 2025. Outcome: factor-box hourly K endpoint hull for July–September 2026. Compare anchor-only, monthly-only, capacity-only, plain anchor-capacity intersection, and the frozen threshold-specific method. Report all months, coverage, widths, and 1/5/10% decisions.

## Interpretation boundary

NWMT is a new target region-month cohort but not a completely unseen BA identity: its 2024–2025 records were in the source calibration pool, although they are excluded from its V121 fold. The date quarter is already observed and target margins are contemporaneous, so this is a retrospective temporal transfer diagnostic, not a month-ahead forecast. The result is U.S. EIA fixed-factor proxy evidence, not China real-CIF validation. Do not tune after scoring; any change requires another holdout.

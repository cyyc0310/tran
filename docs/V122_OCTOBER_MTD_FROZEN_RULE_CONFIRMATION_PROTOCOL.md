# V122 — Frozen Rule on October 2026 Month-to-Date

## Frozen method

Apply the V121 operating rule without tuning:

- τ=600: exact factor-box capacity-constrained set.
- τ=900: anchor∩capacity unless the normalized support gap exceeds the q=20% leave-region-out source boundary, then use capacity only.
- For each target BA in the calibration source pool, exclude that BA from the anchor radius and support-boundary pool. Use the same V120 exact source features and ranks as V121.
- Keep the ±10% thermal EF box, 2025 lagged capacities, exact strict event definition, and ≥80% month-to-date valid-hour eligibility.

## Cohort and window

The candidate pool is the frozen set of all 2026H1 V52 regions with complete-proxy coverage ≥90%, plus NWMT. Include every candidate's October 1 through the last timestamp present in the frozen EIA 2026 Jul–Dec file; selection/eligibility uses only valid-hour counts. Preserve all candidate rows and report months below 80% availability or with infeasible capacities. Do not inspect target K outcomes before cohort/availability freeze.

## Evaluation

For each eligible BA, use its October month-to-date fuel margins and the observed month-to-date hours as the horizon. Report anchor-only, monthly-only, capacity-only, plain intersection, and frozen-rule endpoint coverage, width, and 1/5/10% decisions; stratify τ=900 rows by nonzero factor-box upper support. Use exact rational arithmetic for hourly strict-threshold proxy counts.

## Interpretation boundary

This is a temporal confirmation on a partial month using contemporaneous cumulative margins and hourly labels. It is a nowcast/retrospective identification check, not a month-ahead forecast. The 2026Q3 labels informed V120/V121 development, but October target outcomes have not been scored. U.S. fixed-factor EIA proxies and capacities do not validate real Chinese CIF.

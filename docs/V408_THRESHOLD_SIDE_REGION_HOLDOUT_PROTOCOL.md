# V408: frozen threshold-side energy rule on regions outside the development roster

Frozen: 2026-10-31, before scoring held-out-region outcomes.

## Question

Does the V97/V98 threshold-side monthly energy summary retain its near-sufficiency relative to detailed monthly fuel margins when applied to the 41 EIA balancing authorities excluded from the V90/V97 development roster? This is a held-out-roster replication of a specific identification mechanism, not a claim that these systems were never examined in any other project analysis.

## Frozen roster and period

Use every region in the V52 66-region quality-census pool except the 25-region V90/V97 development roster:

`AECI, AVRN, AZPS, BHBA, CISO, CPLE, CPLW, DEAA, DUK, ERCO, FPC, GLHB, GRIF, GWA, HGMA, HST, IPCO, ISNE, LGEE, NEVP, PACE, PACW, PGE, PJM, PNM, PSCO, SEPA, SIKE, SOCO, SPA, SRP, SWPP, SWPW, TAL, TEPC, TVA, WACM, WALC, WAUW, WWA, YAD`.

Use calendar year 2025 and lagged 2024 EIA-860 capacity. Include every region-month with at least 80% valid hourly proxy rows; the coverage gate is calculated without using the hourly CIF proxy values. Do not select or remove months based on observed threshold counts or capacity agreement.

## Frozen information sets and thresholds

Use the V97 emission-factor proxy and sets unchanged: coal 1000, gas 400, oil 800, other 400, and zero for nuclear, hydro, wind, solar gCO₂/kWh. Test τ=600 and 900; monthly event-frequency targets are K/T≥1%, 5%, and 10%.

- At τ=600, the candidate risk-side monthly field is coal+oil MWh, with within-side EF support [800,1000]; the rest of thermal generation has EF 400.
- At τ=900, the candidate field is coal MWh, with EF 1000; residual thermal EF support is [400,800].
- Add the same lagged coarse technology-capacity envelope (thermal/nuclear/hydro/wind/solar) to detailed-margin and risk-side sets.
- Compare against detailed monthly fuel margins plus the same capacity envelope, and against the no-capacity threshold-side set.

## Scoring and failure handling

The primary endpoint is region-equal mean K-set width and positive/negative/abstention frontier on the candidate information set, compared with detailed monthly margins under the same capacity. Report exact rowwise set containment, observed-K containment, and capacity feasibility rate on the full ≥80%-coverage roster.

If lagged capacity is infeasible for a region-month, do not discard the row: mark the capacity-based set unavailable, report the capacity-free threshold-side set, and count the capacity-dependent decision as abstention. No capacity mismatch may be silently filtered out. If the fixed-factor proxy's observed K falls outside a candidate set, record a failure and do not revise the formula or EF mapping.

Region-months are not independent systems. Report all region-level results and descriptive macro averages; do not use a region bootstrap over 41 systems as confirmatory inference. This is same-source EIA proxy evidence, not direct measured CIF, China validation, or probabilistic coverage.

## Evidence and provenance

Use the V52 quality-census region pool, the V54 native EIA 930 input snapshots, and EIA-860 2024 capacity snapshot. Preserve hashes of the frozen protocol and all inputs in the output manifest. Candidate region labels may be used only to verify set containment and score after the deterministic sets are produced; they do not tune any model, threshold, roster, period, or capacity rule.

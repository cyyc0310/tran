# V226 Frozen confirmation protocol: four-field partial daily snapshots

Protocol frozen before scoring 2026 H2 target outcomes. Date: 2026-10-30.

## Frozen candidate

Use target-month totals for all eight generation categories to compute the monthly generation-weighted fixed-factor anchor. At target inference, keep the center fixed. Randomly sample 12 complete UTC days without replacement; for every hour of each sampled day, observe only four values:

1. hourly total generation,
2. hourly thermal generation total (coal + gas + oil + other),
3. hourly coal generation,
4. hourly gas generation.

The monthly category margins and these observations define per-hour feasible ranges for the unobserved oil and other-fossil values; nonthermal sources have zero EF under the frozen proxy. The sample-day nonconformity is the maximum across its 24 hours of the largest possible absolute deviation from the monthly anchor over the feasible composition set. The 90% split-conformal rank over 12 sampled day scores produces a symmetric interval around the unchanged monthly anchor for every unsampled day. Whole days, not individual hours, are calibration units.

The comparison methods are frozen: (a) the same 12 sampled days with exact eight-category hourly curves and exact daily maximum scores; (b) a source-only interval calibrated from 2025 same-month daily maxima in other BAs, excluding the target BA; and (c) the three-field ablation total + thermal + coal. No fields, sample size, factor values, interval scores, or fallback rules will be changed after viewing target metrics.

## Confirmation cohort and data split

- Source calibration: 2025 native EIA-930 hourly generation.
- Target confirmation: complete months July–September 2026 from the subsequently added native EIA-930 Jul–Dec file. October is excluded because it is incomplete as of this protocol date.
- Include only BAs outside the V224/V225 development roster. This is an outer region and future-time confirmation for the frozen method, not a claim that those BAs have never appeared in any historical project analysis.
- A target BA-month is eligible only if at least 24 complete UTC days remain after the EIA source-quality screen and at least 10 other BAs have usable same-calendar-month 2025 source data. Cohort selection uses completeness and source availability only, never CIF error or interval scores.
- Exclude the target BA’s 2025 history from source calibration. No target BA historical daily/CIF profiles enter the method.

## Outcomes and pass gates

Report macro-BA hourly coverage, simultaneous all-hours-within-day coverage, interval width, Winkler-90, worst BA average simultaneous-day coverage, and paired 95% BA-cluster bootstrap intervals.

The four-field candidate passes this confirmation only if all conditions hold:

1. At least 10 BAs and 30 BA-months enter the cohort; otherwise label the run underpowered.
2. Four-field local calibration reaches at least 90% macro simultaneous-day coverage, and every included BA has at least 90% average simultaneous-day coverage across its target months.
3. Its macro Winkler-90 and width are each below the source-only baseline.
4. Relative to the exact-eight-field local comparator, the four-field method loses no more than 2 percentage points of macro simultaneous-day coverage and has no more than 5% higher width or Winkler-90.

A failure closes this candidate for this cohort; do not retune on confirmation regions. Any later method requires a new protocol and a new unviewed cohort.

## Validity and claim boundary

The fixed factors are the same mechanical proxy used in development, and the labels derive from EIA hourly fuel generation, not metered emissions. Random-day coverage is a design-based marginal statement over the predeclared sampling randomization; it is not fixed-sample, simultaneous-month, or region-conditional coverage. This is a retrospective month-after reconstruction because target monthly fuel margins are required.

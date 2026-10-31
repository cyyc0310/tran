# V223 Sparse local daily fuel snapshots reduce interval inefficiency

Date: 2026-10-30. Run: `results/v223-sparse-daily-risk-snapshots-r1/`.

## Main result

On 75 held-out countries/aggregates and 1,800 complete 2024–2025 country-months, source-only leave-country-out conformal calibration achieved macro-country coverage **90.94%**, mean width **80.71**, and Winkler-90 **132.81 gCO2/kWh**.

| Local daily snapshots | Macro coverage | Mean width | Winkler-90 | Difference in coverage vs source-only (95% country bootstrap CI) | Difference in Winkler vs source-only (95% country bootstrap CI) |
|---:|---:|---:|---:|---:|---:|
| 7 | 100%, unbounded | ∞ | ∞ | +9.06 pp [6.47, 11.83] | undefined |
| 8 | 100%, unbounded | ∞ | ∞ | +9.07 pp [6.46, 11.94] | undefined |
| 9 | 90.32% | 62.14 | 75.60 | −0.62 pp [−3.21, +2.19] | −57.28 [−74.59, −44.66] |
| 12 | 92.56% | 65.83 | 75.07 | +1.62 pp [−0.97, +4.50] | −57.62 [−74.53, −45.18] |
| 18 | 94.89% | 70.16 | 75.31 | +3.96 pp [+1.36, +6.77] | −57.46 [−74.97, −44.72] |
| 21 | 91.17% | 62.44 | 73.65 | +0.21 pp [−2.41, +3.02] | −58.99 [−76.35, −46.33] |

At 9 snapshots, **34 of 75 countries** had empirical mean coverage below 90%, despite macro coverage being close to nominal. At 12, no country fell below 90% when averaging over its 24 target months and repeated random samples; this is a descriptive result for this benchmark, not a finite-sample country-conditional guarantee. The interval width is non-monotone with budget because the conformal rank is discrete and scores vary by sampled days.

**China subset (24 months):** the source-only radius had 100% empirical coverage, width 81.36, Winkler 81.36. Twelve random daily fuel snapshots gave 92.31% coverage, width 26.72, Winkler 36.27. Nine gave 89.94%, width 23.03, Winkler 36.06. These unusually strong source-only China results reflect very smooth secondary-data labels; they are not independent validation of Chinese hourly CIF.

## Interpretation

V222 found no point-prediction gain from transferring weekday shapes. V223 instead finds a useful information-acquisition candidate: keep the monthly generation-weighted center fixed, use a small random sample of local daily fuel mixes to calibrate only the radius, and avoid overfitting the center. Nine observations are the rank threshold for a finite nonrandomized 90% radius; 12 is the smallest tested budget with macro coverage above 90% and no below-90% country average in this dataset. Neither number is established as the real-world minimum.

The gain is potentially useful for the paper’s information-budget framing, but **not an algorithmic novelty claim**. Design-based conformal prediction already studies finite-population coverage under sampling designs, and recent work studies conformal prediction regions for trajectories observed at sparse random times. Our use of a simple random within-month sample and fixed monthly CIF anchor is narrower than those general methods. The contribution would have to be the CIF-specific identification/value frontier and a sampling contract that survives independent operational validation. Moreover, China daily labels in this source are estimated through a process that uses daily coal consumption; the test does not prove that 12 publicly available or independently measured Chinese daily snapshots exist or suffice.

## Novelty check

Design-based conformal inference already covers finite-population sampling designs, including simple random sampling. A 2026 preprint on conformal regions for continuous-time trajectories handles sparse random observation times and coverage between samples using trajectory regularity assumptions. V223 uses neither a new conformal guarantee nor a new generic sparse-trajectory method. Its possible contribution is empirical and task-specific: quantify which local fuel fields and how many random target-day snapshots are needed to calibrate CIF risk when only the monthly generation anchor is otherwise known.

References: [design-based conformal prediction](https://www150.statcan.gc.ca/n1/pub/12-001-x/2023002/article/00007-eng.htm), [conformal regions for randomly sampled trajectories (2026 preprint)](https://arxiv.org/abs/2608.29559).

## Decision

Continue with this as a measurement-design hypothesis, not a frozen method. Before making a paper claim, repeat on an independent source with direct daily/hourly fuel generation or CIF labels, preregister the random-day sampling design, report country-conditional failures, and compare against at least one baseline that uses target daily information to update the center. If such data cannot be obtained, present this as a secondary-data feasibility probe and retain the conclusion that monthly statistics alone do not identify daily risk.

## Files

- [Protocol](V223_SPARSE_DAILY_RISK_SNAPSHOT_PROTOCOL.md)
- [Runner](../scripts/v223_sparse_daily_risk_snapshots.py)
- Artifacts: `results/v223-sparse-daily-risk-snapshots-r1/`

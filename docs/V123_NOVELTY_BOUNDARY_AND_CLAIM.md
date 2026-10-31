# V123 — Novelty Boundary After the Frozen October Confirmation

## What the experiment now supports

On a frozen October 2026 month-to-date cohort, an exact capacity-feasible set was paired with a cross-region calibrated monthly anchor. At τ=900, a pre-frozen support-gap rule replaced the intersection with the physical set when local coarse inputs showed support conflict. This recovered the one observed intersection miss (SC), covered 37/37 capacity-feasible factor-box proxy endpoints, and retained a mean interval width 51% below capacity-only. The window contained only three nonzero τ=900 support rows; all were covered, and every output abstained on positive risk.

This is an encouraging application-level mechanism result. It is not yet evidence of a new general conformal theorem or an actual Chinese CIF prediction model.

## Related work changes the novelty claim

- **Conditional and shift-aware conformal prediction already target heterogeneous coverage.** Gibbs, Cherian, and Candès formulate finite-sample guarantees over classes of covariate shifts and subgroups, while noting the limits of universal conditional coverage ([JRSSB paper](https://doi.org/10.1093/jrsssb/qkaf008)). Recent localized conformal work also studies conditional coverage and oracle interval efficiency ([2026 preprint](https://arxiv.org/abs/2608.06206)). Thus, “use a local risk feature to adapt conformal width” is not a sufficient novelty claim.
- **Conformal uncertainty combined with physical safety constraints already exists.** SafeConPhy uses conformal bounds to certify physical-system control outputs ([NeurIPS SafeGenAI Workshop paper](https://openreview.net/pdf?id=OkPDLLNLnM)). A generic “conformal set plus physical fallback” is therefore not novel by itself.
- **Hourly electricity emissions factors already combine temporally granular generation with annual factors.** IEA's Real-Time Emissions Factors product provides up-to-hourly power-sector intensity/emissions and combines generation data with annual national-statistics factors ([IEA product page](https://www.iea.org/data-and-statistics/data-product/real-time-emissions-factors-3)). This weakens any broad claim that reconstructing hourly average intensity from lower-frequency generation and annual factors is new.
- **Average and marginal intensity are distinct estimands.** Existing regression and forecast work targets marginal emissions; for example, the CAISO neural-network study forecasts day-ahead hourly marginal factors ([Mayes et al., 2024](https://doi.org/10.1016/j.jclepro.2023.139895)). The paper must say clearly that this project targets the hourly count/set of a generation-weighted average intensity proxy, not marginal emissions.

## Defensible contribution candidate

The strongest defensible framing is an **information-budget benchmark for hourly average-CIF tail-risk partial identification**, with an adaptive support-aware backoff rule:

1. Define a public-data contract (monthly fuel margins, annual/lagged group capacity, EF uncertainty, optional local hourly data).
2. Compute an exact physically feasible set for the count of hours above a threshold under that contract.
3. Use a region-level calibrated anchor to reduce the set only while its observable risk-support gap is within the calibration domain; otherwise revert to the physical set.
4. Evaluate a coverage–width–decision frontier by information budget and tail-support stratum, including the abstention cost.

The likely novelty is the **task-specific formulation, sharp feasible-set engine, and support-aware information-budget evaluation for high-threshold hourly average-emissions risk under sparse public statistics**. The fallback itself is a promising design pattern, not a claimed new statistical principle.

## A useful guarantee, with assumptions made explicit

Let `Y` be the true proxy K endpoint for a held-out region block, `A` the calibrated anchor interval, and `C` the physical feasible set. If the exact identification assumptions ensure `Y ∈ C` and region-level split conformal ensures `P(Y ∈ A) ≥ 1−α` for a matching exchangeable target block, then `A ∩ C` covers `Y` whenever `A` does, and replacing the intersection by `C` on any feature-selected subset cannot reduce coverage. The union of those two outputs therefore retains at least the anchor's marginal guarantee.

This observation is not a new theorem; it clarifies the method's safety argument. The current experiment does not yet verify the assumptions: the V120 score takes a maximum over 2024–2025 months while V122 scores one partial October block; target completeness selection can alter exchangeability; capacities and fuel margins are proxy inputs; and no Chinese hourly emissions labels are present. A paper must use matched calibration blocks and audit these assumptions before invoking a formal guarantee.

## Minimum work before claiming a paper breakthrough

- Recalibrate with source scores matched to target horizon/season, then evaluate on multiple untouched quarters and multiple spatially new balancing areas. Report high-threshold nonzero-support months separately; three October examples are too few.
- Compare against localized/weighted conformal, Mondrian/group conformal, direct quantile regression, historical-profile and persistence anchors, monthly-only sharp sets, capacity-only sharp sets, and the plain intersection, at matched information budgets.
- Show both width and decision utility: coverage, interval score/size, positive/negative certificate rates, abstention, and conditional coverage in nonzero-tail-support strata.
- Stress capacity vintage error, fuel-margin reporting error, EF ranges, missingness, and target data-selection rules.
- Obtain same-boundary Chinese hourly generation and emissions (or validated fuel-consumption/CEMS-derived emissions) before making a China CIF performance claim.

## Current verdict

The October result is the first credible **method candidate** that both repaired a seen high-threshold support failure and retained a substantial width gain in a new time window. It does not yet prove nontrivial risk prediction, broad regional transfer, statistical novelty beyond related conformal and safety-set methods, or real Chinese CIF validity. The next research gate is matched-window calibration and an untouched full-quarter test with substantially more nonzero tail support.

# V401 protocol: hourly-total value for tail-risk identification from monthly generation margins

## Question

How much can China-compatible monthly generation categories constrain hourly generation-CIF tail counts, and how much additional identification comes from knowing the target's hourly total generation? This is a set-identification and value-of-information experiment, not a point-forecast method.

## Target and information sets

- Use the seven EIA balancing authorities and 2024 held-region/month cells already used in V369/V395. Results remain development evidence because these labels have been viewed before.
- Define a China-compatible fixed-factor generation proxy as `700 g/kWh × hourly thermal generation / hourly total generation`; thermal is coal+gas+oil and all other reported categories have zero proxy intensity. This deliberately aggregates the thermal category and is not measured emissions or official CIF. EIA's extra `other` category is retained as a separate source residual and assigned zero proxy intensity, not merged into China's solar category.
- **M:** use only each month’s five coarse generation margins (thermal, hydro, nuclear, wind, solar). EIA's separate `other` margin remains a nonthermal residual in the source data. M's analytical sharp count bounds use the monthly energy-weighted proxy mean and an unconstrained positive hourly-total profile: for a mixed thermal/nonthermal month and threshold below 700, the count lies in `[1,H]` when the monthly mean exceeds the threshold, and `[0,H−1]` otherwise. At threshold 700 the count is structurally zero under the fixed factor; degenerate all-thermal or all-nonthermal months also have point bounds.
- **M+T (oracle):** add the true hourly total generation series, but still hide the hourly fuel allocation. Solve for the minimum and maximum number of hours above 300, 500, and 700 g/kWh over all nonnegative hourly fuel allocations satisfying both the monthly class margins and each hour's total. `M+T` is an information-value upper bound, not a deployable input under the user's current China data contract.
- **M+T+D (oracle):** retain hourly totals and add each UTC day’s true thermal generation margin, while leaving its within-day hourly allocation unknown. This isolates how much daily local fuel reporting could add beyond hourly totals alone; it is also not in the current China data contract.
- Hold the proxy factor fixed at 700 g/kWh for the primary optimization, so the width isolates uncertainty in hourly dispatch composition. Thermal-factor support uncertainty is reported as a limitation, not mixed into the main width.

## Optimization

For each region-month and threshold, use a mixed-integer linear program. The model allocates hourly thermal generation; nonthermal generation is each known hourly total minus thermal. Binary variables indicate whether threshold intensity is strictly exceeded. Constraints enforce exact monthly thermal totals, optional per-hour total balance, optional UTC-day thermal margins, nonnegative generation, and the linear threshold inequality. Compare M+T and M+T+D under identical monthly margins and hourly totals. Report solver status and gap. A fixed numerical tolerance defines strict crossing consistently.

## Scoring and interpretation

Report the observed fixed-factor proxy tail count, M, M+T, and M+T+D identification bounds, width, feasibility of the observed count, and normalized width. Aggregate first within month and then equally by region. Report each BA, especially MISO, PACE, and LGEE if present in this panel; do not claim independent confirmation or a conformal guarantee. Feasibility proportions are consistency checks, expected to be one by construction rather than predictive coverage estimates.

If `M+T` remains wide, hourly totals alone do not resolve dispatch-driven tail risk. If `M+T+D` narrows further, daily thermal margins are a candidate lower-cost local statistic; if it remains wide, finer local temporal dispatch information is needed. Neither oracle result is deployable accuracy. The interval coverage is a feasibility check and is expected by construction because the source hourly allocation satisfies the margins. This experiment pertains only to generation-side CIF; it says nothing about consumption-side CIF, whose cross-region flows and source intensities remain separate inputs.

## Reproduction

Runner: `scripts/v401_monthly_margin_tail_identification.py`. Corrected run output is written to a new `results/v401-monthly-margin-tail-identification-r7/` directory and refuses overwrite. Earlier scratch outputs used a broad `[0,H]` M baseline, incorrect equal-region aggregation, merged the source-only `other` category into solar, or missed the structurally zero threshold control and are not evidence.

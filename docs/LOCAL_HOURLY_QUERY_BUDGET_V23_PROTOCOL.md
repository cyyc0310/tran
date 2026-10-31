# V23: source-only information value of sparse local hourly CIF queries

Registered 2026-10-27 before parsing the registered source-day fuel values. This is an adaptive development diagnostic after V19/V20/V22. It is not an independent confirmation, a forecast experiment, or evidence that these measurements are available in China's monthly-only information budget.

## Question and information classes

For a completed historical UTC day, can a small number of exact or 10 g/kWh-interval hourly CIF observations reduce uncertainty about the strict exceedance count `K_tau = sum(1[C_h > tau])`, after giving every method the same contemporaneous six-hour `H/G` and all-24-hour positive `G` oracle information? The base information is deliberately generous and unavailable in the user's stated China budget. Sparse hourly CIF is an additional explicit local-label budget; fuel telemetry that can reconstruct it is also a carbon-derived local label.

The experiment measures past-path identification only. It does not estimate future event probabilities, improve a forecast, or establish transport to a new region. Its source files are reused development data. Exact calculations are exact relative to decimal strings in those files and the registered fixed-factor proxy, not certified physical measurements.

## Fixed cohort and estimands

- Inherit the exact seven-source V20 cohort and fixed full-day mask. Select the first and last valid day within each of the seven source regions using the inherited V20 day grid: 14 days per region, 98 source-days total. No day or region is selected by response value.
- The primary strict threshold is `tau=300 g/kWh`; `tau=100` is a prespecified sensitivity. `tau=600,900` are excluded from the empirical query-efficacy claim because the parent cohort has no observed exceedance at those thresholds.
- For each day, the base set is four exact six-hour block totals `H_b=sum(G_h*C_h)`, block totals `G_b`, and all 24 contemporaneous positive `G_h`, with `C_h in [0,1000]`. This is V20's fixed-G sharp `HG/support` identification class, applied blockwise.
- Add `b in {0,1,2,4}` queried hourly CIF labels. Query hours are selected without access to unqueried labels. `exact` observations reveal the decimal CIF value; `10g_interval` observations reveal the enclosing closed 10 g/kWh interval, intersected with `[0,1000]`.

## Frozen query policies

At each budget, compare four deterministic policies:

1. `uniform_clock`: nested fixed schedules, b=1 `{12}`, b=2 `{6,18}`, b=4 `{3,9,15,21}`.
2. `seeded_random`: one nested four-hour schedule per source-day generated from SHA256(`region|day|V23`) and NumPy PCG64; the seed depends only on the public key, never labels.
3. `widest_feasible_CIF`: greedily query the unqueried hour with the widest marginal CIF interval implied by the base block H, fixed G, and prior observed query intervals. Ties go to the lowest UTC hour.
4. `one_step_minimax_Kspan`: greedily choose the query whose worst possible post-query K-set span is smallest. For exact labels, enumerate every interval endpoint and all status-feasibility breakpoints induced by the at-most-six-hour block subset inequalities, plus one rational midpoint per open cell. For 10g intervals, enumerate all feasible 10g bins. Ties go to the lowest UTC hour. This is a one-step lookahead heuristic, not a globally optimal active-query policy.

The same response information is supplied to the identification solver and to each policy after it is observed. Policies may use previous queried responses only. A queried exact C is fixed; a quantized response constrains C to its reported interval. For every block, enumerate all event/non-event assignments of its six hours and retain assignments whose weighted-energy interval contains H, respecting strict `C>tau` and interval endpoint openness. Combine feasible block counts by Minkowski sum. The resulting K-set must contain the realized K for every source-day, policy, budget, threshold and precision mode.

## Metrics and decision rules

- Primary: per-source rate of **correctly certified** `K_tau=0` and `K_tau>=1`, and unresolved rate, averaged over its 14 fixed days, then macro-averaged over seven sources. A certificate is issued only if every K in the feasible set has the same event status; wrong certificates are a hard failure.
- Secondary: same rates for `K_tau>=6`; mean and median remaining K-span; exact versus quantized sensitivity; source-specific outcomes and true positive/negative day counts.
- Keep every source/day result. No hour/day is an independent region. No strategy or threshold is selected after seeing outcomes.
- The information-selection signal passes this development gate only if, at b=4, one adaptive policy reduces source-macro unresolved rate by at least 10 percentage points versus the best of the two non-adaptive policies in **both** exact and 10g-interval modes, with zero wrong certificates and zero excluded true paths. This does not authorize a forecasting claim or a new-region opening; it only earns a separately frozen follow-up.

## Provenance and limits

Read only the seven source files already hash-bound by the completed V20 parent and only timestamps in its sealed 1000-day mask. No target roles, unseen regions, PACE/LGEE, later source response, new prediction model or parameter fitting. Do not alter V20/V22 files or results. Bind this protocol, new producer, source/parent hashes and outputs in a new result directory; refuse overwrite. The work uses hourly fuel files as an isolated source-history oracle to simulate extra local measurements. It is outside the current China monthly/yearly-only deployment budget, has no real publication-vintage audit, and cannot be counted as a China feature.

Interpretation boundary: a positive result would show only that selected historical CIF labels can reduce ambiguity about past K under a generous hourly-G/H base. It would not show these labels are affordable, available before a forecast, useful for future K probabilities, or a novel active-learning algorithm. A negative result rejects only these four policies and this fixed small cohort/information class.

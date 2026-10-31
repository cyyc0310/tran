# V140 — Conditional Tail-Risk Learning with Complete Monthly Fuel Margins

## Purpose

V139 improves average width but misses rare high-tail systems. Test whether complete monthly composition and seasonal context predict the remaining hourly tail-count position, using strong conditional quantile baselines and a physical-coordinate variant. The user's confirmed deployment budget is complete monthly generation categories. Lagged coarse capacity is an optional research budget, not a confirmed Chinese input.

## Frozen development design

- Reuse all 589 V139 region-months and robust K endpoint labels, two thresholds, and the V120 ±10% EF contract. Retain every month, including capacity-unavailable cases.
- Build only month-level features by aggregating the same audited hourly snapshots. Compare a monthly-only budget with a monthly + lagged coarse-capacity budget. For monthly-only, obtain the exact monthly-margin endpoint hull at EF low/high; for capacity budget, use V139's audited capacity hull with [0,T] fallback.
- Common features: eight monthly fuel shares, log(1+monthly MWh), month sine/cosine, factor-box monthly anchor endpoints, and the chosen budget's physical lower/upper/width fractions. Optional capacity features: each coarse group's capacity share and bounded utilization M/(M+TC), plus the input-only capacity-unavailable flag. No region identifiers, target-hour shapes, weather, or past target K are predictors.
- Outer region-LOO over all 25 regions. For each held-out region, split the remaining 24 regions into 12 fit and 12 calibration regions by a deterministic SHA-256 ordering, separately for seeds 41 and 83. The same split applies to all methods and budgets. Test region, fit regions and calibration regions must be disjoint. Both repetitions are reported; do not choose the better one.
- Alpha values: 0.10 and 0.20. Calibration uses maximum endpoint error over all months within a source calibration region, then finite-sample rank ceil((n+1)(1−alpha)). Source-month sample weights give equal weight to each fit region.

## Same-budget methods

1. Physical hull, with availability fallback.
2. Monthly-constant anchor with absolute-error group calibration and physical clipping.
3. Monthly-constant anchor with physical-width normalized group calibration.
4. Conditional quantile boosting for K_low/T and K_high/T, fitted on all fit months; calibrate absolute endpoint error after physical projection.
5. The same direct conditional model fitted only on input-nonsingleton physical months, to control for removal of trivial cases.
6. Conditional quantile boosting for (K_low−L)/(U−L) and (K_high−L)/(U−L), fitted only on nonsingleton fit months; group-calibrate normalized endpoint error and map back through the target physical hull.

Use sklearn GradientBoostingRegressor with quantile loss, lower/upper quantiles 0.05/0.95, 150 estimators, learning rate 0.05, max depth 2, minimum leaf size 15, fixed model seed 20261028. Sort predicted endpoints and clip them into the physical hull before calibration. Nonnegative conformity scores permit expansion only. Physical singletons return their exact physical value. Outward integer rounding and final physical clipping apply to all methods.

## Reporting and audits

Save the input feature panel, raw-source hashes, capacity hashes, feature list, split memberships, calibration radii and per-month predictions. Assert exact month/hour denominator match to V139, no target/fit/calibration overlap, valid physical containment, finite features, nonempty interval subset, alpha nesting, and complete scoring denominator.

Report each repetition separately, including region-equal monthly coverage/width, simultaneous all-month regional coverage, coverage conditional on K_high>0 and K_low>0, ≥1/5/10% positive/negative/abstain/certificate errors, and a fixed-90%-penalty robust interval score: width +20(lower−K_low)_+ +20(K_high−upper)_+. This endpoint-stress score is not CRPS and does not establish hourly forecast accuracy. Report PACE/LGEE only in subsequent transfer probes, not by relabelling the source cohort.

## Decision boundary

Normalized conformal scores, CQR, interval-valued conformal inference and temporal disaggregation already exist. An efficiency result here is a mechanism probe, not sufficient originality. A candidate must improve rare-tail coverage and interval efficiency against direct same-budget conditional models in both source splits before considering a freeze and genuinely unopened confirmation. All current source outcomes were previously examined. Target-month aggregates make this reconstruction/risk identification; prospective forecasting additionally requires forecasting those aggregates with their publication delays.

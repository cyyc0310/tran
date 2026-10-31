# V336 protocol: inner-CV shrinkage of local heat-rate transfer

Registered 2026-10-30 before running. V335 found a small Ember-source increment from local heat rates over mix-only ridge-NNLS, but the evidence was weak for the NBS source and only seven outer region clusters are available. V336 tests whether choosing how much local efficiency information to transfer using training regions can preserve positive cases while avoiding negative transfer.

## Frozen model and folds

Reuse the V335 whole-region outer folds, years, annual MEE targets, lagged NBS/Ember five-group mixes, two CEC rate definitions, and nested ridge-NNLS. For each source/rate, replace the thermal share by:

`thermal_share × [1 + gamma × (local_rate/300 − 1)]`

with frozen `gamma ∈ {0, .25, .5, .75, 1, 1.25}`. Gamma 0 is exactly the source's mix-only model; gamma 1 reproduces V335. For every outer fold, jointly choose gamma and ridge λ from `{0, .001, .01, .1, 1, 10}` by pooled inner whole-region MAE, using only outer-training target labels. Ties choose the smaller gamma, then smaller lambda. Keep full gamma=1 and gamma=0 as diagnostic arms.

## Outcomes and gate

Report pooled and yearwise MAE/RMSE/P90, region-year scores, selected gamma/λ, and paired seven-region bootstrap intervals for nested-selected minus mix-only and nested-selected minus fixed-full-rate model. This is a candidate adaptive-transfer increment only if (a) the nested model improves over gamma 0 in both years, (b) improves in at least 4/7 regions, (c) paired region-cluster interval is below zero, and (d) yearwise P90 does not worsen materially. No region-specific rule may use its outer labels.

## Limits

This is an exploratory annual MEE-factor transfer test, not hourly CIF evaluation or interval coverage. It has 7 outer region clusters, 2 target years, retrospective yearbook inputs with unresolved as-of release dates, and no user's source table header/vintage audit. Nested tuning is not a novelty claim by itself.

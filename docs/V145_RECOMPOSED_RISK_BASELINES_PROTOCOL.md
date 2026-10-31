# V145 — Recomposition Risk Responses inside Strong Same-Budget Baselines

## Purpose and status

Test ordinary fuel recomposition and conditional risk heads before proposing a special mechanism. Primary development cells are LGEE/600 and PACE/600, where V143 found large reducible interval excess. SC and threshold 900 remain in the full report. All identities, 2024–25 months and outcomes were already examined. No genuine confirmation is opened.

Use V144's exact literal factor contract and immutable source-profile response cache. Refit every baseline under that contract. Report the six one-hour label changes separately; do not compare numerical-contract changes as method improvements. Monthly-only local features and source-hourly data are the allowed budget; no capacity, weather, target-hour observations or oracle factor-floor features.

## Frozen region roles

Targets: PACE (18 months), LGEE (24), SC (24). Seeds 41/83, alpha 0.10/0.20. Reuse the original 25 source identities and 18 eligible added identities. For each target, remove its identity, order original source identities by SHA-256(seed/target/source), use the first 12 as calibration, and the rest plus eligible additions as fitting. This gives exactly 30 fit and 12 calibration regions. Assert equivalence to V140's SC roles and V142's PACE/LGEE roles.

For every calibration/target query, response donors come only from the 30 fit identities. For proper-fit feature rows, remove the query's whole region from the donor bank, giving a region-out-of-fold teacher; the final prediction function uses the full 30-fit bank for both calibration and target. Report the 29-versus-30 teacher change. Never use calibration/target profiles as response donors, and save donor IDs per query. Synthetic responses do not increase calibration region count.

## Ordinary response distributions

For each donor region select the two eligible months nearest the query calendar month by circular month distance, then chronological month and profile ID for ties. Use all valid hours of each complete donor profile in the cached operator. Equal-region prior assigns 1/R mass to each region, split equally over its two profiles.

Compare pooled weights with ordinary composition-conditional weights obtained by multiplying the prior by exp(−Hellinger_squared/0.3²) and normalizing. Hellinger_squared is 0.5 sum_f(sqrt(query_share_f)−sqrt(donor_share_f))². Do not tune the bandwidth. Report unsupported-fuel query mass and effective donor-region count for both weighting rules.

Initial response bounds are the weighted 0.05 quantile of low-factor response fractions and the weighted 0.95 quantile of high-factor response fractions. Point offsets are the separate weighted medians. Response-derived features, for each weighting rule, are low/high mean, median and standard deviation; initial low/high quantiles; weighted unsupported query mass; and effective donor-region count. These use only query monthly margins and permitted source profiles.

## Same-budget comparison

1. Physical monthly-margin hull.
2. Fixed monthly anchor calibrated over all 42 allowed sources; no fitted stage.
3. Fixed monthly anchor calibrated over the 12 nearest source-region mean fuel-share vectors, as in V142.
4. Direct CQR on all fit months, using V140's common monthly-only feature list.
5. Direct CQR on input-nonsingleton fit months, same common features.
6. Direct CQR with the common features plus both sets of response-derived features.
7. Residual CQR with the same expanded features, fitting K_low/T−conditional_low_median and K_high/T−conditional_high_median, then restoring the offsets at query time.
8. The same residual CQR fitted on input-nonsingleton months.
9. Pooled recomposition initial bounds with global endpoint calibration.
10. Conditional recomposition initial bounds with global endpoint calibration.
11. Conditional recomposition with standard normalized endpoint calibration. Its scale is max(low-factor response SD, high-factor response SD, 1/query_hours), so dispersion of dispatch responses and the gap due to the factor contract are explicit separate quantities.

The CQR models retain V140's quantile loss, 0.05/0.95 heads, 150 estimators, learning rate 0.05, max depth 2, minimum leaf 15 and seed 20261028. Equal total weight per fit region, recomputed after active-month filtering. Sort/physically project initial endpoints, then calibrate a nonnegative maximum endpoint error score. For methods 4–10 use tail-fraction error; method 11 divides by its declared scale. First take maximum over months within each of the 12 calibration regions, then finite-sample rank ceil((n+1)(1−alpha)). Anchors use their stated source pool. Outward integer rounding and physical clipping apply throughout.

## Reporting and gate

Save features, exact donor IDs, fixed roles, radii, predictions and hashes. Assert donor exclusion, no label columns in features, complete 66-month target denominator, valid physical subsets, alpha nesting, and all cache consistency checks. Report every method, target, seed, threshold and alpha: endpoint coverage/width, robust interval score, certificate rates/errors, and source-support diagnostics. The factor-floor accounting is evaluation-only.

Fuel recomposition, conditional donor weighting, residual CQR and heteroscedastic conformal scores are existing techniques. This matrix can show where a new mechanism is needed; winning this baseline comparison alone does not establish originality. No method is frozen for unopened confirmation and no hourly forecast/CRPS result is claimed by this monthly-tail experiment.

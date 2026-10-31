# V141 — Source Diversity with Frozen Monthly-Only Targets and Calibration

## Question

V140's source pool contains one strongly coal-dominated region, SC; 20/24 of its monthly coal shares exceed every other source's maximum. Test whether adding source histories from already examined region identities improves transfer, and distinguish additional source count from coal-regime coverage. This is a source-data ablation, not an originality claim.

## Frozen data and roles

- Keep V140's original 25 held-out region identities, all 589 target region-months, 2024–25 period, two thresholds (600/900), two source-role seeds (41/83), alpha values (0.10/0.20), and its exact saved 12-fit/12-calibration region roles.
- Local input budget is full monthly generation by the eight EIA categories. Use the V140 monthly-only feature list, physical monthly-margin hull, quantile models, region-equal fit weights, group-max calibration score, outward integer rounding and endpoint-stress metrics unchanged. Lagged capacity is not an input in this round.
- Candidate additions are V118's already scored region identities outside the original 25: AVRN, AZPS, CISO, CPLE, DUK, ERCO, FPC, IPCO, LGEE, PACE, PNM, PSCO, SOCO, SPA, SRP, TAL, TEPC, TVA, WALC. No new region identity or future period is opened for confirmation.
- Build additional donor monthly features and source-only factor-box K endpoints from the same 2024–25 raw snapshots and EIA proxy quality definition. A month needs at least 80% complete UTC hours. A donor needs at least 12 eligible months. Save every candidate's availability, including exclusions. Source selection uses availability and monthly fuel composition only, not K.
- Separate input preparation from model scoring, and save an immutable donor panel with raw hashes, factor-vector representation and quality audits so model iterations do not reload/relabel raw data.

## Controlled source scenarios

1. Original V140 models and anchor baselines: reuse their saved monthly-only predictions without refitting.
2. Add all eligible additional donors to the original 12 fit regions.
3. Add coal-heavy donors: eligible donors whose average eligible-month coal share is at least 0.40.
4. Add an equal number of lower-coal donors, chosen by fixed SHA-256 ordering of seed/target-region/donor-region. If either category is smaller, use the same minimum category count in scenarios 3 and 4; apply the fixed ordering within both categories. Report exact rosters and region/month counts. Scenario 2 still includes every eligible donor.

All three expanded scenarios retain exactly the same original calibration regions and target months. Additions never enter calibration. Fit models give equal total weight to each source region, so additional months do not implicitly upweight a region.

## Models and safeguards

Expanded scenarios fit the three V140 conditional models: direct CQR on all fit months, direct CQR on input-nonsingleton fit months, and physical-coordinate CQR on nonsingleton fit months. Monthly-only anchor baselines remain the original saved ones. Hyperparameters and feature columns are copied exactly from V140; no target score selects a model or donor.

For source label construction, represent the V120 floating factor-vector values as exact decimal rationals, as in its saved source labels. Integer-valued generation rows can use exact integer excess coefficients with arbitrary-precision integer products; noninteger rows use rational arithmetic. Check both paths against direct rational strict-threshold counts on the first eligible month of each candidate region and synthetic threshold-boundary examples. Save checks and fail on mismatch. These checks establish implementation equivalence under the assumed EF contract, not physical EF validity.

Assert role separation, donor identities within the declared already examined roster, complete original target denominator, unchanged original/calibration rows, valid interval/physical containment, alpha nesting, and exact reproduction of archived original predictions. Report all failures, all scenarios and both seeds. Report SC explicitly, plus conditional coverage for K_high>0 and K_low>0, overall coverage/width, region-level simultaneous coverage, positive/negative/abstain/errors and the fixed-90%-penalty robust interval score.

## Interpretation gate

A donor expansion gain is an information-support result, not a new prediction mechanism. If source support repairs SC but ordinary CQR matches the proposed physical-coordinate method, retire normalization as the main innovation. If all variants still fail, investigate hidden hourly dependence under similar monthly composition instead of adding arbitrary calibration widths. Any future confirmation remains frozen only after a distinct, attributable method increment; this opened-pool study does not qualify as blind validation or as hourly-CIF forecast accuracy.

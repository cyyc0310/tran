# V141 — Appropriate Source Support Repairs Rare-Tail Transfer; Normalization Is Not the Increment

## Controlled result

The original 589 target months, 25 held-out identities, both source-role seeds and all original calibration regions remain fixed. Among 19 already examined additional donor candidates, 18 have at least 12 eligible historical months; WALC has 11 and is excluded from fitting. The additions supply 347 eligible region-months. The coal-heavy group, selected using mean monthly coal share ≥0.40, contains LGEE and PACE. Equal-count lower-coal controls use two donors under the frozen hash selection.

All 80 source-count arithmetic checks pass. The scoring run contains 70,680 rows, including exactly reproduced V140 original predictions and 42,408 new rows. Role, interval, denominator and alpha-nesting audits report zero failures.

At alpha=0.10, ordinary direct CQR fitted on all fit months gives:

| Threshold | Source scenario | Macro coverage, seeds 41 / 83 | Width, seeds 41 / 83 | Coverage where K_high>0, seeds 41 / 83 |
|---:|---|---:|---:|---:|
| 600 | Original sources | 92.95% / 96.00% | 176.04 / 190.14 h | 79.48% / 88.76% |
| 600 | Add all eligible sources | 98.48% / 98.82% | 178.17 / 177.33 h | 95.39% / 96.40% |
| 900 | Original sources | 97.33% / 97.67% | 62.53 / 116.28 h | 79.83% / 87.83% |
| 900 | Add all eligible sources | 99.50% / 99.83% | 33.90 / 62.36 h | 91.13% / 99.13% |
| 900 | Add only two coal-heavy sources | 99.50% / 99.50% | 34.80 / 39.75 h | 72.00% / 72.00% |
| 900 | Add two matched lower-coal sources | 97.33% / 97.67% | 61.41 / 117.67 h | 79.83% / 87.83% |

Compared with the unchanged monthly anchor, adding all donors to direct CQR reduces τ=900 width by 32.4% / 28.2% while increasing overall coverage from 98.00% / 98.50% to 99.50% / 99.83%. Its robust endpoint interval score improves from 90.43 / 124.46 to 37.80 / 62.66. At τ=600, the expanded direct model is about 10% narrower than the anchor but slightly less covered, so there is no all-threshold dominance.

The anchor uses the same unchanged calibration role in this causal source-fitting ablation; it is not an independently optimized baseline consuming every newly added source label for calibration. A later method comparison must permit strong baselines to use the whole allowed source budget appropriately.

## What happened in SC

For direct CQR at τ=900, SC coverage rises from 41.67% to 95.83% in both repetitions when all additions enter fitting. Its own width increases from 79.92 h to 342.79 / 308.58 h; other regions become narrower enough that the aggregate width still falls. This is selective redistribution of uncertainty, not uniform interval expansion.

Adding just LGEE and PACE gives SC 100% τ=900 coverage but widths 572.08 / 544.50 h. Adding two lower-coal sources leaves SC at 41.67%. Thus source regime matters in this tested design. Region count is matched, but eligible month counts and other source properties are not identical; do not attribute the entire difference solely to coal share.

The two-coal-donor direct model's conditional coverage for all nonzero-tail regions is only 72%, despite its strong overall coverage and repaired SC. The active-month control does better on this conditional metric. This is another reason to retain the full method frontier rather than choose a scenario from SC alone.

At τ=600, SC's repaired 100% coverage comes with an approximately full-month interval (725 h average). The positive-tail identification problem is therefore not solved there.

## Method decision

The physical-coordinate variant does not establish a distinct advantage over ordinary same-budget CQR. At τ=900 and seed 83, expanded direct CQR is both narrower and more covered than the coordinate variant. Retire normalization as the proposed main innovation. The demonstrated increment is source-support coverage, which is valuable benchmark evidence but not an original algorithm.

Next, assess PACE and LGEE as held-out development targets under the complete monthly budget, removing each target's own history from fitting and calibration. Compare ordinary CQR with anchors that are also allowed to consume the expanded source budget. Only then decide what remaining dispatch-risk mechanism requires innovation. No genuinely unopened confirmation region or future period has been opened, and the original hourly forecast/CRPS gate remains unevaluated by this monthly-tail experiment.

## Reproduction

- [Frozen protocol](V141_SOURCE_DIVERSITY_ABLATION_PROTOCOL.md).
- [Source preparation](../scripts/v141_prepare_expanded_sources.py), immutable input panel in `results/v141-source-expansion-input-r1/`.
- [Scoring runner](../scripts/v141_source_diversity_ablation.py), all scenarios and both seeds in `results/v141-source-diversity-ablation-r1/`.

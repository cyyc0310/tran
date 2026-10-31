# V226 Frozen four-field confirmation passes on unseen BAs and future months

Date: 2026-10-30. Protocol hash is recorded in `results/v226-frozen-four-field-confirmation-r1/metadata.json` and matches the frozen document.

## Cohort

The frozen confirmation used 26 BAs outside the V224/V225 development roster and 74 eligible BA-months from July–September 2026. October was excluded as incomplete. The method had no target-BA historical CIF/daily curve in its source calibration; source-only scores came from 2025 data in other BAs. Cohort inclusion used native-record completeness and source availability, not CIF errors.

This is an outer region/time confirmation for V225's method. It does not claim that every selected BA is new to all earlier TransCIF experiments.

## Frozen results

| Method | Macro hourly coverage | Simultaneous day coverage | Width | Winkler-90 |
|---|---:|---:|---:|---:|
| Source-only, target BA excluded | 95.32% | 82.72% | 299.49 | 342.89 |
| 12-day exact eight-field local calibration | 99.00% | 93.32% | 263.05 | 266.32 |
| 12-day frozen four-field local calibration | 99.09% | 93.82% | 272.49 | 275.59 |
| 12-day three-field ablation | 99.18% | 94.42% | 321.09 | 323.97 |

The four-field method's simultaneous day coverage exceeded 90% in every BA; the worst was CPLE at 91.18%. Relative to source-only calibration, four-field Winkler improved by 67.30 [BA bootstrap 95% CI 18.48,116.999], and simultaneous day coverage rose by 11.10 points [0.88,22.93]. Width fell by 27.00, but its CI [−95.90,+44.30] crosses zero.

Relative to exact eight-field local calibration, the four-field method had 0.50 points higher simultaneous day coverage, 3.6% greater mean width, and 3.5% greater Winkler. Paired BA-bootstrap CIs for the small width and Winkler increases were [+0.14,+27.53] and [+0.10,+27.19]. It therefore passed all frozen gates: enough BAs/months, all BA mean day coverage above 90%, both macro efficiency measures below source-only, and no more than 5% loss in width or Winkler versus exact eight fields.

Some BAs still need wide intervals: the four-field mean width was 560.98 in PACE and 665.13 in PNM. The aggregate gain does not imply uniformly narrow intervals. Source-only day coverage was particularly poor in PACE (3.1%).

## What this confirms

The field-compression signal was not limited to the 24 development BAs or 2026H1. Under a frozen protocol, hourly total + thermal + coal + gas on 12 random full days retained nearly the same interval efficiency as exact eight-fuel hourly snapshots, while outperforming source-only calibration on macro Winkler and day coverage.

This is the strongest candidate breakthrough so far: a **monthly-margin-conditioned, partially observed risk envelope** that turns a few aggregate local readings into a conservative day-level CIF interval. The new scientific content is the fuel-conservation feasible set and its measured-field frontier, not conformal prediction by itself.

## Limits and next confirmation

- Labels are EIA fuel-generation proxies under fixed factors, not measured emissions.
- The exercise is post-month reconstruction; the target month's eight margins must be available.
- It still requires 12 days × 24 hours × 4 local aggregate fields. The user's currently described public monthly tables do not supply these hourly snapshots.
- Some selected BAs have been used in earlier TransCIF work; this is unseen only relative to the V224/V225 development roster and 2026H2 outcome window.
- A China claim requires either an independent local hourly-generation/CIF source or a clearly stated additional-data acquisition route. No blind China result exists.
- Related literature already covers partially labelled conformal calibration and robust calibration with score upper bounds; novelty must be argued specifically for the physical CIF identification set and field-budget frontier.

## Files

- Frozen protocol: [V226 protocol](V226_FOUR_FIELD_PARTIAL_RISK_FROZEN_PROTOCOL.md)
- Runner: [confirmation script](../scripts/v226_frozen_four_field_confirmation.py)
- Valid artifacts: `results/v226-frozen-four-field-confirmation-r1/`

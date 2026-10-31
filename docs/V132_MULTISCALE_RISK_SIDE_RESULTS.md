# V132 — Three-Day Risk-Side Reporting Captures Most Daily Information Gain

## Results

The V129 frozen cohort produced 358 region-months, 21 regions, and 8,592 scored rows across two thresholds, three tail-rate decisions, and four temporal bins. The 48-case synthetic cross-check matched the closed-form risk-side set to the general V129 solver; full-data nesting checks had zero failures.

Equal-region mean results at the ≥1% decision target:

| Threshold | Bin | Mean reporting scalars/month | Risk-side width (h) | Detailed-fuel oracle width (h) | Negative / positive / abstain |
|---:|---:|---:|---:|---:|---:|
| τ=600 | Monthly | Existing monthly contract | 316.08 | 315.97 | 56.01% / 0% / 43.99% |
| τ=600 | 1 day | 90.44 | 225.16 | 222.28 | 56.01% / 0.71% / 43.28% |
| τ=600 | 3 days | 31.69 | 246.10 | 245.10 | 56.01% / 0.48% / 43.52% |
| τ=600 | 7 days | 14.86 | 268.76 | 268.31 | 56.01% / 0% / 43.99% |
| τ=600 | 14 days | 8.86 | 289.67 | 289.40 | 56.01% / 0% / 43.99% |
| τ=900 | Monthly | Existing monthly contract | 213.50 | 213.38 | 70.35% / 0% / 29.65% |
| τ=900 | 1 day | 90.44 | 192.76 | 189.42 | 70.35% / 0% / 29.65% |
| τ=900 | 3 days | 31.69 | 197.95 | 196.77 | 70.35% / 0% / 29.65% |
| τ=900 | 7 days | 14.86 | 202.63 | 202.08 | 70.35% / 0% / 29.65% |
| τ=900 | 14 days | 8.86 | 206.54 | 206.22 | 70.35% / 0% / 29.65% |

Each block reports three values: thermal total, threshold high-side generation, and clean total. A three-day contract uses about 31.7 values per month, 65% fewer than daily reporting. It retains 77% of the daily interval-width reduction at τ=600 and 75% at τ=900. Its robust interval is only 1.00 h and 1.18 h wider than the same-bin detailed-fuel oracle. Seven-day reporting retains about 52% of the daily width benefit with fewer values, while 14-day reporting retains about 29–34%.

The decision impact remains modest. At τ=600, three-day reporting yields 0.48% positive certificates at the ≥1% target and lowers abstention by 0.48 percentage points relative to monthly information. At τ=900 it does not change the positive/negative/abstain rates at this target. The principal measured benefit is a narrower identified set, not broadly improved tail classification.

## Interpretation

This makes a **three-day, threshold-indexed summary** a useful candidate information contract for the next confirmation: it appears to retain most of the daily information gain at roughly one-third of the reporting volume. This is an exploratory operating point selected on the already-open V129 cohort; it must be frozen and validated on a new time block before claiming transfer.

The broad premise that temporal aggregation loses decision-relevant information is established in electricity-system modeling, and time-resolved (including quarter-hourly) electricity emission-factor accounting has been published. The plausible novelty is therefore narrower: a factor-support-aware, threshold-specific *partial-identification* curve that measures which local reporting aggregates reduce hourly tail-count ambiguity and selective abstention under a fixed information budget. The three-day result alone is not a methodological novelty claim. See [Merrick on temporal representation in electricity planning](https://stanford.edu/~jmerrick/temporal_resolution.html) and [Kühne et al. on time-resolved electricity emission factors](https://link.springer.com/article/10.1007/s11367-023-02240-3).

The novelty bar is higher because the [AAAI-26 multi-frequency CIF forecasting paper](https://ojs.aaai.org/index.php/AAAI/article/download/41310/45271) already uses wavelet-based multi-frequency features and dynamic cross-variable relationships, evaluating hourly CIF forecasts across four Australian electricity markets. That work assumes rich time-series inputs and pursues point prediction; our candidate studies what can be *identified* from coarse local aggregates with no local hourly CIF history. A paper must make that information-contract distinction explicit and compare against low-frequency disaggregation and probabilistic forecast baselines wherever their input budgets are matched.

The experiment is retrospective oracle coarsening of U.S. EIA hourly generation, with fixed-factor proxy labels and sensitivity EF intervals. It excludes capacity from every arm and does not establish Chinese availability, release timing, error rates, or actual CIF coverage. The user-reported Chinese monthly categories are not present as an original source file in this workspace; the local China panel is a secondary Ember estimate, not a direct hourly-label validation source. A separate Chinese data-contract audit is still required before claiming this three-day summary is obtainable or legally publishable.

## Reproduction

- Protocol: [V132 protocol](V132_MULTISCALE_RISK_SIDE_PROTOCOL.md)
- Script: [V132 runner](../scripts/v132_multiscale_risk_side_information.py)
- Outputs: `results/v132-multiscale-risk-side-r1/`
- Protocol SHA-256: `24932a337a537ebbcb5977e7cbdd9eb6a2052eb3e669f94032b6ef6b0de1ed70`
- Script SHA-256: `9b7b4f1674aa78ec3e3449dbc894305b32a19a161288c4635c1ba67577076f8f`
- Summary SHA-256: `27f2a26aafdcda1e4afd6989b22f6c302a056a0b0d7e07cbb1ae2bafd6a2ebd2`

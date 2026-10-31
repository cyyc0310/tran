# V126 — ±20% Fuel-Factor Uncertainty Stress Results

> V139 correction notice: the source support-calibration routine calls the formerly unguarded capacity solver. Its handling of monthly-energy-infeasible source inputs needs re-auditing before the archived mechanism results are reused. See [correction scope](CAPACITY_CONTRACT_CORRECTION.md).

## Main result

The frozen support-gap rule retained endpoint coverage when the coal/gas/oil/other factor box widened from ±10% to ±20%, but its incremental coverage benefit disappeared in this October rerun. Of 44 candidate regions, 40 met the 80% hourly-availability rule and 37 also passed the lagged-capacity feasibility check.

| Threshold | Method | Coverage | Mean width (hours) |
|---|---|---:|---:|
| τ=600 | Capacity-only frozen rule | 37/37 | 263.4 |
|  | Plain anchor∩capacity | 37/37 | 195.0 |
| τ=900 | Capacity-only | 37/37 | 222.0 |
|  | Plain anchor∩capacity | 37/37 | **120.8** |
|  | Frozen q=20% support-gap rule | 37/37 | 144.2 |

At τ=900 the frozen rule is 35.0% narrower than capacity-only, but 23.4 hours wider on average than plain intersection. It falls back for JEA and PACE; neither fallback repairs an endpoint miss in this stress run. Six of 37 factor-box target sets have a nonzero upper tail count (JEA, LGEE, NWMT, PACE, PNM, SC), and all six are covered by the plain intersection as well as the frozen rule. At τ=600 the previously frozen capacity-only policy gives no width reduction relative to the physical set.

Relative to V122's ±10% run, the τ=900 capacity width rises from 183.2 to 222.0 hours and frozen-rule width rises from 89.8 to 144.2 hours. The fraction narrower than capacity drops from 51.0% to 35.0%. Thus the width gain survives this sensitivity expansion, but weakens materially. The wider factor box also makes the plain intersection cover 37/37, whereas it covered 36/37 under ±10%; this means the gate's October coverage-repair claim is not robust to the factor-box choice.

## Scope correction for the EF stress

The runner evaluates three EF vectors: all fossil factors jointly at −20%, nominal, and all jointly at +20%. This is a **one-dimensional common-mode multiplier stress**, not a box in which coal, gas, oil, and other-fossil factors vary independently. The original protocol wording called it a fuel-factor “box” and was too broad. The numbers above are retained as the run actually performed; they do not establish robustness to independent factor uncertainty. A separate factor-wise stress must be run before making that broader claim.

The follow-up is now complete: [V127](V127_INDEPENDENT_EF20_CORNER_RESULTS.md) evaluated all 16 independent corners, and [V128](V128_CONTINUOUS_EF_BOX_SCORE_BOUND_RESULTS.md) added a conservative calibration-score bound for the continuous box. Both retain the same October frontier, but the gate adds no observed coverage over plain intersection in this sensitivity cohort.

## Interpretation

This is a useful robustness result, not a new independent confirmation: V126 reuses the October labels already opened in V122 and V125. It supports the claim that a factor-robust monthly anchor intersected with a capacity-feasible set can remain substantially narrower than capacity-only under this proxy cohort. It does **not** show that the support-gap fallback adds coverage under ±20%; in the observed target cohort it adds width without changing endpoint coverage.

The current method is therefore best described as a conservative support-aware interval-selection rule with a sensitivity-dependent efficiency/coverage tradeoff. The key scientific risk is whether its support boundary transports across genuinely unseen regions and full future months. The data still come from U.S. EIA hourly generation and lagged capacity, with fixed factor perturbations; ±20% is a stress scenario, not an empirically justified Chinese factor interval.

## Reproduction

- Frozen protocol: [V126_EF20_ROBUSTNESS_PROTOCOL.md](V126_EF20_ROBUSTNESS_PROTOCOL.md).
- Runner: [v126_ef20_robustness_stress.py](../scripts/v126_ef20_robustness_stress.py).
- Row-level target results, availability denominator, source scores/gaps and input hashes: `../results/v126-ef20-robustness-stress-r1/`.

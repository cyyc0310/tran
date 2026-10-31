# V103: the coarse-capacity K-set remains informative under a ±10% EF stress box

## Question and setup

V102 established an exact K identified set under fixed fuel EFs. V103 tests whether that information value collapses when all thermal-fuel factors are uncertain by ±10% around the project's fixed proxy values. On the same 513 monthly-energy-feasible V100 region-months, use exact lower-factor, nominal-factor, and upper-factor capacity-constrained sets at τ=600/900. The endpoint envelope is `[min(K_low-EF), max(K_high-EF)]`.

For any fixed hourly generation allocation, hourly intensity is coordinatewise nondecreasing in every fuel EF. Therefore the lower-EF exact set supplies a sharp minimum K over the factor box, and the upper-EF exact set supplies a sharp maximum K. The reported interval is the hull of these sharp endpoints; this does not prove that every interior integer is feasible under a single globally fixed factor vector. The ±10% box is a stress scenario, not an empirical or official China factor uncertainty interval.

## Region-equal results

| τ | Set | Mean width (h) | Positive ≥1% | Negative ≥1% | Abstention | Nominal proxy K contained |
|---:|---|---:|---:|---:|---:|---:|
| 600 | Nominal EF exact set | 224.45 | 7.28% | 53.94% | 38.78% | 100% |
| 600 | ±10% EF robust hull | 235.83 | 4.36% | 53.94% | 41.70% | 100% |
| 900 | Nominal EF exact set | 136.32 | 0.00% | 67.11% | 32.89% | 100% |
| 900 | ±10% EF robust hull | 152.52 | 0.00% | 67.11% | 32.89% | 100% |

The width increase is about **11.4 h** at τ=600 and **16.2 h** at τ=900. The 155 months that fail V100's target-hour peak-capacity screen remain in this cohort. The nominal proxy K is contained in every robust interval; all 513 month-level checks pass.

## Interpretation

This is a useful robustness result for the exact partial-identification mechanism: in this EIA proxy cohort, moderate multiplicative uncertainty in fuel factors slightly widens capacity-constrained K sets but does not remove their negative-certification value at high thresholds. At τ=900, factor uncertainty also does not create positive tail certificates. That is an honest asymmetric result: the information contract can rule out many high-tail months, but it does not identify positive high-tail risk at this threshold under the current factor box.

This is not evidence that real fuel-factor uncertainty in China is ±10%, that the proxy is true CIF, or that a future region will receive calibrated coverage. The actual emission-factor intervals, generation/consumption boundary, fuel classification, capacity mapping, and release vintage must be sourced before interpreting this as a deployment guarantee.

## Reproducibility

- Protocol: `docs/V103_FACTOR_UNCERTAINTY_SHARP_SET_PROTOCOL.md`
- Script: `scripts/v103_factor_uncertainty_sharp_set.py`
- Frozen output: `results/v103-factor-uncertainty-sharp-set-r1/`
- Protocol SHA-256: `287aaf97403a378e7bc79d092ccdcfe49a05f5f3efc4214bf3c10deb503a9abc`
- Script SHA-256: `a1cffa4f9782c826c5060082f9c69e3ff71ed41cb15bf14bc4d88b2b1accc29a`
- Summary SHA-256: `d85cabf5a32698f8aaa6662eda60a9b3d1598975d86aed89cdc96f12e83c9792`

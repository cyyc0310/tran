# V143 — Factor Ambiguity and Dispatch-Transfer Excess Are Different Optimization Targets

## Accounting result

The diagnostic uses 43 already examined region identities, 936 region-months, two thresholds and 73,032 saved prediction rows. All physical-floor, label-join, unchanged-prediction and oracle-repair invariants pass. No predictor is fitted or calibrated.

For a fixed observed hourly allocation and the assumed factor box, define F=K_high−K_low. Any interval containing both endpoint responses must have width at least F. This is a necessary width bound for that contract, not a distribution-free forecasting theorem and not an oracle input available at a monthly-only target.

| Region | Threshold | Mean oracle factor floor | Mean monthly physical width | Physical excess above factor floor |
|---|---:|---:|---:|---:|
| LGEE | 600 | 0.00 h | 720.00 h | 720.00 h |
| LGEE | 900 | 651.42 h | 720.96 h | 69.54 h |
| PACE | 600 | 199.78 h | 730.11 h | 530.33 h |
| PACE | 900 | 2.39 h | 729.67 h | 727.28 h |
| SC | 600 | 71.29 h | 725.00 h | 653.71 h |
| SC | 900 | 190.13 h | 725.21 h | 535.08 h |

At LGEE/900, the factor-contract floor is about 90.4% of observed hours. The nearest-source anchor's 707.38 h width is 55.96 h above the floor, leaving at most a 7.9% mean-width reduction if every endpoint remains covered on these same months. A large reduction there would require changing the factor-information contract or accepting misses. This statement uses the full-coverage sample oracle; it is not a lower bound for procedures permitted to miss selected months.

At LGEE/600, F is zero while the CQR intervals remain 720 h wide. This has a much larger potential for reducing dispatch-transfer uncertainty, provided a method can justify the contraction using permitted source/local data. PACE has a different pattern: its high-threshold oracle factor floor is only 2.39 h, so hundreds of hours of local-anchor width are mostly excess; the all-source anchor and one CQR split are much more efficient but retain small misses or split sensitivity.

## Too narrow versus misplaced

Every LGEE/900 direct-CQR miss in V142 has predicted width below its corresponding oracle factor floor: 87.5% of months in seed 41 and 54.17% in seed 83. The problem is not just moving an otherwise sufficiently wide interval. The oracle expansion would raise those models' mean widths to 673.29 / 688.04 h, still below the fully covered nearest-source anchor, but the expansion uses forbidden target responses and is diagnostic only.

The diagram shows how the assumed factor box creates different floors across the two thresholds. Purple LGEE months are close to a full-month floor at 900 and zero at 600. This does not imply that coal share alone determines dispatch risk or that the assumed factor box is empirically calibrated for China.

![Oracle factor-contract floor against monthly coal share](../results/v143-factor-floor-diagnostic-r1/factor_floor_by_coal_share.png)

## Consequence for the next method round

Maintain two explicit components: uncertainty from the factor contract and error in transferring the hourly dispatch response. Test composition-to-tail response transfer within the established fuel-recomposition architecture, against ordinary recomposition, direct CQR, CQR with the same derived features, and all-source/local anchors. Existing recomposition and normalized/conformal calibration are baselines, not independently original mechanisms. A candidate must reduce the dispatch excess while retaining factor endpoint coverage, especially at LGEE/600 and PACE/600, before reopening confirmation.

The original goal remains trustworthy hourly CIF prediction or a transferable benchmark under limited legal inputs. These monthly-tail diagnostics clarify a bottleneck; they do not satisfy the hourly coverage/CRPS or novelty gates.

Reproduction: [protocol](V143_FACTOR_FLOOR_DIAGNOSTIC_PROTOCOL.md), [runner](../scripts/v143_factor_floor_diagnostic.py), full row accounting and [PDF figure](../results/v143-factor-floor-diagnostic-r1/factor_floor_by_coal_share.pdf) in `results/v143-factor-floor-diagnostic-r1/`.

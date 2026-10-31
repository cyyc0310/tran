# V104 results: unseen-region factor-robust confirmation

## Outcome

The frozen V103 exact-set method was applied to five region codes absent from prior region-specific `docs/` and `scripts/` analysis, selected only from the V52 quality census: CPLE, FPC, IPCO, PNM, SOCO. All five had 2026 complete-proxy coverage ≥98%; all 30 region-months (Jan–Jun 2026) passed the 80% month coverage and lagged-2025 group-energy feasibility checks. The test is region-heldout within a previously observed time period, not a prospective time holdout.

At each month, the exact monthly-margin + five-group-capacity K set contains the observed fixed-EF proxy K. The ±10% thermal-factor endpoint hull also contains all 30 observed Ks at both thresholds. This confirms the mathematical containment behavior out of region on the proxy labels.

## Selective decision frontier

Macro averages give the same means across 1%, 5%, and 10% targets because the identified K sets are broad relative to the thresholds:

| Threshold | Nominal mean width | ±10% robust mean width | Robust positive | Robust negative | Robust abstain |
|---|---:|---:|---:|---:|---:|
| 600 gCO₂/kWh | 410.3 h | 434.1 h | 0% | 0% | 100% |
| 900 gCO₂/kWh | 252.4 h | 270.0 h | 0% | 20.0% at 1%/5%; 23.3% at 10% | 80.0% at 1%/5%; 76.7% at 10% |

At τ=900, the nominal set gives 33.3% negative decisions at the 10% target, but the robust hull retains only 23.3%; factor uncertainty turns some apparent negatives back into abstentions. There are no positive certificates at either threshold. Region heterogeneity is substantial: IPCO is fully negative at τ=900 because its proxy generation has little/no high-EF thermal support, while the other regions mostly abstain; FPC and PNM have particularly wide sets.

## Interpretation

The result supports a narrow claim: exact capacity-constrained partial identification and the monotone EF-box hull transfer across five unseen regions without proxy-label set failures. It does not support a claim of useful general CIF prediction: the selective frontier is mostly abstention, and coverage is only a logical set-containment check against a fixed-factor proxy, not probabilistic calibration or actual-CIF validation. The 2026H1 period itself was seen in earlier work, so only region novelty is clean.

The next required comparison is the same region-months and labels under (a) monthly fuel margins alone and (b) monthly margins plus lagged group capacity. Without this paired baseline, the unseen-region result does not establish that capacity caused the few useful negative certificates or the narrower sets. The prospective research gate remains unmet until the method has a useful nontrivial decision frontier and an actual Chinese hourly CIF validation channel.

## Reproducibility

- Frozen protocol: `docs/V104_UNSEEN_REGION_ROBUST_CONFIRMATION_PROTOCOL.md`
- Runner: `scripts/v104_unseen_region_robust_confirmation.py`
- Outputs: `results/v104-unseen-region-robust-confirmation-r1/`
- Runner and protocol hashes are recorded in `summary.json`.

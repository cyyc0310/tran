# TVA 2026H1 frozen confirmation: asymmetric candidate failed

Run date: 2026-10-27. Frozen method and target choice: [V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md](V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md). The first label-reader correction was registered before the successful scoring pass in [V50_TVA_CONFIRMATION_BOUNDARY_AMENDMENT.md](V50_TVA_CONFIRMATION_BOUNDARY_AMENDMENT.md). Machine-readable record: results/disjoint-calibration-v50-tva-confirmation/score_seal.json.

## Target and sealing

TVA was the seventh qualifying region in the earlier metadata-only V6 quality screen, reserved by its fixed first-six selection rule. The fixed 2026H1 test is one region and 181 complete UTC days. The target interval predictions were generated from the seven frozen V50 outer point/risk functions and sealed before the scoring phase. V50 source models, ten development calibration sources, the V6 audit, raw snapshot inputs, monthly context, target prediction grid and sealed predictions are hash-recorded. No target fit, target calibration, threshold tuning or source substitution was performed.

The first scoring command stopped at the exact support check because it read only native snapshot 8 and obtained 180 days. It created no metric table or score. V6's frozen audit assigns the 2026-01-01 boundary day to snapshot 7 and the remaining target days to snapshot 8. The pre-scoring amendment fixed the reader to those two hash-pinned snapshots. This initial attempt did computationally open target rows during label construction; we disclose that access rather than call the labels technically unopened. No score or interval comparison was exposed in that failed pass. The corrected scorer then verified exactly 181 days and produced the sealed result.

## Fixed comparison

| Method | 90% coverage | Width | Interval score | Lower miss | Upper miss |
|---|---:|---:|---:|---:|---:|
| Frozen candidate: EF shift + MCStd + separate tails | 77.67% | 191.77 | 294.46 | 22.33% | 0.00% |
| Same EF shift + MCStd + symmetric calibration | 95.30% | 207.63 | **221.83** | 4.70% | 0.00% |
| Same EF shift + HETScale + symmetric calibration | 100.00% | 338.29 | 338.29 | 0.00% | 0.00% |
| Same EF shift + CQRplus + symmetric calibration | 100.00% | 403.95 | 403.95 | 0.00% | 0.00% |
| Simple B0 anchor ×2 | 90.72% | 245.40 | 269.85 | 0.00% | 9.28% |

The candidate shrank the matched symmetric interval width by 7.6%, but coverage fell 17.6 percentage points and interval score worsened 32.7%. Its score was also 9.1% worse than the simple factor-2 anchor, which met the 90% marginal coverage threshold in this one region. The candidate's entire failure was in the lower tail; no upper-tail misses occurred. This reverses the development pattern where separate tails repaired WACM's upper misses. The same-information symmetric MCStd baseline is the best of the five on interval score and maintains 95.3% observed coverage.

The month pattern sharpens the diagnosis:

| Month | Candidate coverage | Symmetric MCStd coverage | Candidate lower miss |
|---|---:|---:|---:|
| Jan | 97.18% | 100.00% | 2.82% |
| Feb | 78.87% | 96.13% | 21.13% |
| Mar | 80.11% | 97.04% | 19.89% |
| Apr | 63.89% | 82.08% | 36.11% |
| May | 89.65% | 99.19% | 10.35% |
| Jun | 55.28% | 97.08% | 44.72% |

This is consistent with a lower-tail threshold that does not transfer over time in the held region; it is not evidence that a post-hoc TVA-specific correction should be fitted. We did not tune q values, change the calibration pool, or rerun a modified candidate on TVA.

## Scientific decision

The candidate is rejected for transfer. Its development gains on ten reused sources did not survive the one preselected new-region test. Generic asymmetric conformal therefore does not constitute a defensible TransCIF algorithm contribution here; the result also warns that source-pooled tail allocation can be region- and time-sensitive even after a monthly level correction. The symmetric same-information method is more reliable on this one region but its 82.08% April coverage prevents claiming seasonally stable calibration. The strong HETScale/CQRplus controls over-cover with much wider intervals.

This is one single-region, single-semester failure, not evidence that every new region will fail. It is still a valid negative confirmation and must remain in the paper/benchmark record. We do not use its 4.7% symmetric lower-miss result to reselect or tune a new TVA method. Further method development must return to development data or acquire a new, independently held region; no remaining untouched region is established in this workspace after TVA scoring. China transfer is untested, the EIA target is a fixed-EF generation proxy, and the monthly release date is an artificial month-end-plus-30-day budget.

## Reproduction artifacts

Prediction seal: results/disjoint-calibration-v50-tva-confirmation/prediction_seal.json (TVA_PREDICTION_SEALED_LABELS_UNOPENED); score seal: results/disjoint-calibration-v50-tva-confirmation/score_seal.json (TVA_CONFIRMATION_SCORED). Aggregate and monthly tables are adjacent CSVs. The two-stage predictor and frozen scorer are scripts/confirm_v50_tva_asymmetric.py and scripts/score_v50_tva_asymmetric.py. V50 and V6 source artifacts were not modified.

# V50 nested monthly-q shrinkage falls back to the global threshold

Sol, 2026-10-27. This is a source-held-out comparison of a standard nested-shrinkage control, not an original method. It tests whether blending a monthly-mix-predicted source p90 score with a global equal-source 90th-percentile threshold can repair the undercoverage of the direct conditional-q method.

## Frozen design and audit

The [protocol](V50_NESTED_MIX_Q_SHRINKAGE_PROTOCOL.md) held out one of ten entire external sources at a time, within each of seven frozen V50 outer functions. For every outer fold, a second leave-one-source-out loop used only the nine outer-training sources to select a threshold blend. The fixed grid was `{0,.25,.50,.75,1}`. A blend could be selected only if inner source-macro coverage was 90–95%, its worst inner source was at least 85%, and then it had the lowest inner interval score; otherwise the rule fell back to global `lambda=0`.

The producer verified V50 seals, all 70 external prediction archives, ten source label files, full-mix features, and monthly input hashes. It refit no production point/risk head and read no target, PACE/LGEE, 2025 response, or unseen source. The completion receipt and four output CSV hashes are recorded in `results/disjoint-calibration-v50-nested-mix-q-shrinkage/completion.json`.

## Results

The inner coverage gate failed for every candidate in all 70 outer×held-source folds. The selected lambda was therefore 0 in **70/70** folds; nested shrinkage exactly returned the global equal-source MCStd threshold.

| Method | Source-macro coverage | Width | Interval score | Lower miss | Upper miss |
|---|---:|---:|---:|---:|---:|
| Global MCStd | 86.86% | 623.18 | 833.66 | 0.00% | 13.14% |
| Nested mix-q shrinkage | 86.86% | 623.18 | 833.66 | 0.00% | 13.14% |
| Direct monthly conditional q | 77.13% | 493.57 | 892.10 | 2.93% | 19.94% |
| Global HETScale | 87.30% | 641.59 | 883.97 | 0.07% | 12.63% |
| Raw anchor ×2 | **88.50%** | **528.53** | **712.72** | 0.01% | **11.50%** |

The nested selector did what it was designed to do—reject unsupported local threshold adaptation—but it produced no gain over the global calibrator. The direct conditional rule remains narrower but substantially undercovers and has a worse interval score. This is evidence against spending more effort on local-q shrinkage with this monthly proxy and these ten sources; it is not evidence that all local calibration is impossible.

## Research conclusion

The monthly mix level predicts source risk summaries somewhat better than a constant source mean, but that signal is not reliable enough to set a target-specific interval threshold. Cross-fitting and a stringent training-only shrinkage rule avoid the development leakage and undercoverage shortcut, yet simply return the global method. The best observed held-source interval comparator remains the simple factor-2 anchor; no method satisfies the earlier source coverage/efficiency gate in this comparison.

This is a negative method result on ten repeatedly used development sources. It does not validate Chinese transfer, and the feature's simulated EIA-style release timestamps and fixed project EFs are outside the user's China validation budget. No new source was opened or method frozen.

# Monthly mix does not predict held-source tail direction

Date: 2026-10-27. Frozen design: [V50_TAIL_DIRECTION_INFORMATION_PROTOCOL.md](V50_TAIL_DIRECTION_INFORMATION_PROTOCOL.md). Output: results/disjoint-calibration-v50-tail-direction-information/summary.json.

## Result

The probe used the already computed ten-source held-out metrics for B0-SizeET-AlignedKeep + MCStd + EF-only correction + symmetric calibration. The response was each source's average lower miss minus upper miss across the seven fixed outer functions. It reused only monthly input tables and date positions from the sealed calibration metadata; it did not reopen hourly labels, prediction archives, or any target/unseen data. Region was the unit (n=10).

| Fixed one-feature model | LOO MAE: null → feature | LOO RMSE: null → feature | Correct tail-imbalance sign |
|---|---:|---:|---:|
| Three-month EF proxy | 0.05394 → 0.05289 | 0.07011 → 0.07083 | 4/10 → 4/10 |
| Fossil share | 0.05394 → 0.06014 | 0.07011 → 0.07412 | 4/10 → 4/10 |
| Wind+solar share | 0.05394 → 0.06595 | 0.07011 → 0.09492 | 4/10 → 4/10 |

The EF feature changes MAE by only 0.00105 (about 1.9%) while worsening RMSE and leaving sign accuracy unchanged. Fossil and wind+solar shares worsen both errors. None of the three coarse features demonstrates usable out-of-source information about which tail to tighten.

## Interpretation

This is a small descriptive diagnostic on ten reused development regions, not a significance test and not proof that all monthly information is uninformative. It does, however, remove the simplest proposed rescue for the TVA failure: current monthly mix summaries do not reliably tell us which target tail is unsupported. Applying source-level asymmetric shrinkage from these features would add complexity without evidence that it can preserve coverage.

The research bottleneck is now sharper: a new-region method needs either (a) a pre-target observable that directly constrains local hourly risk/tail imbalance, or (b) an explicitly conservative way to retain uncertainty when no such information exists. A subsequent experiment should quantify the value of candidate local measurements under a fixed information budget and compare them against the month-only B0 anchor on development sources. Any such result remains development-only until one or more new regions are prospectively held out.

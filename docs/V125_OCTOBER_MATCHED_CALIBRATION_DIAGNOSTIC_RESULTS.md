# V125 — October-Matched Calibration Diagnostic Results

## Main finding

Recalibrating on source-region October records only collapsed the τ=900 conformal radius to exactly zero for all 37 capacity-feasible V122 targets. The source Octobers had no region-level score above the 90% order statistic, while the new target month contained three nonzero factor-box upper-support rows. Used by itself, this matched-season anchor missed JEA (K upper 3 h) and SC (162 h). LGEE's October monthly mean already exceeded τ=900, giving a wide anchor interval that covered its 502 h upper endpoint.

The frozen q=20% support-gap rule identified JEA and SC as out-of-support and fell back to their exact capacity sets. The combined rule then contained 37/37 factor-box endpoints. Mean τ=900 width was 62.4 h, compared with 183.2 h for capacity-only and 89.8 h for the V122 full-year-calibrated frozen rule. The rule emitted 89.2% negative certificates and 10.8% abstentions, with no positives; all covered proxy endpoints were consistent with those decisions.

At τ=600, the October-matched radius averaged 0.314 versus 0.528 for the full-year score. The plain anchor-capacity intersection covered 36/37 capacity-feasible rows; the frozen method retained the capacity-only set, covering 37/37 at mean width 253.5 h.

## Interpretation

This diagnostic sharpens the mechanism: season-matched conformal calibration can collapse when the calibration season contains almost no tail residuals; the support-gap gate detects a subset of target months where the monthly mean under-represents physically possible high-threshold hours and lets the exact feasible set take over. That is a useful failure-aware backoff result, not a guarantee that q=20% is universally optimal. The check reuses V122 labels and calibration blocks are full historical Octobers while the target is partial October, so it cannot serve as independent validation. The small number of nonzero tail-support rows remains the main limitation.

## Reproduction

- Protocol: `V125_OCTOBER_MATCHED_CALIBRATION_DIAGNOSTIC_PROTOCOL.md`.
- Runner: `../scripts/v125_october_matched_calibration_diagnostic.py`.
- Summary and rows: `../results/v125-october-matched-calibration-diagnostic-r1/`.

# V50 fossil-role increment shrinkage path

Registered 2026-10-27 before computation. The prior low-dimensional composition ablation found that adding fossil share to the fixed-EF proxy improved average level scores but failed interval coverage, particularly for SC. This test shrinks only the incremental fossil-share term toward the EF-only model.

## Fixed design

- Reuse the same V50 seals, legal three-month as-of fuel shares, ten sources, seven outer functions, 90-day eligible labels, Scale-anchor paths, and factor-2 intervals. Recheck all frozen hashes; no target/unseen source or point/risk refit.
- Within each outer × held-source fold, on the other nine sources fit a source-equal weighted EF-only intercept/slope model for daily mean residual. Standardize EF and fossil share from training rows only. Residualize the standardized fossil-share feature against the weighted EF-only design (intercept and EF); regress the EF-only training residual on that orthogonal fossil component. This yields an incremental fossil-role correction that is zero when the fossil term has no conditional support.
- Apply the fixed path `EF-only prediction + λ × fossil-role increment` for λ∈{0, .25, .50, .75, 1.00}. This is equivalent to shrinking the extra composition coefficient while keeping the EF baseline fixed. No fold-wise or held-source-specific coefficient selection.
- Compare all fixed λ to raw anchor and simple equal-source global bias. Report per-source and macro level/CRPS/coverage/width/interval-score/tail misses/clipping. Source remains the unit.

## Interpretation boundary

The λ curve is exploratory development evidence on ten reused sources. No best λ may be presented as confirmatory from these folds. A selected rule would require a frozen nested source-only tuning procedure and then genuinely unseen-source validation. The analysis cannot establish China transfer, causal dispatch effects, or method novelty.

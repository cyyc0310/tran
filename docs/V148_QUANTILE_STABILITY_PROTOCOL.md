# V148 — Quantile-boosting numerical stability and corrected strong controls

Frozen after V146/V147 development and a targeted V142→V145 discrepancy investigation. The diagnostic already observed PACE/900 seed41 library-default widths of 17.5 versus 599.94 hours under reordered identical fitting rows. With initial 95% quantile zero, the installed sklearn 1.8.0 loss assigns the same negative gradient to zero and positive responses; tree impurity roundoff can then induce arbitrary splits. A 1e-10 split-gain guard has already been tried on that single diagnostic fold and yields a constant model. Neither of these observations is an independent validation result.

## Frozen comparison

Keep canonical V144 inputs and V145's two source rosters, 30 fitting / 12 calibration regions, PACE/LGEE/SC months, alpha .10/.20, same common and source-response feature budgets. Compare all-month direct-common CQR, direct-response CQR and response-residual CQR. No new region or time is opened.

For each model compare three explicitly named implementations:

1. `library_default`: exact V145 implementation.
2. `split_guard`: same loss, with `min_impurity_decrease=1e-10` to reject roundoff-only splits.
3. `strict_tie_guard`: same split guard and pinball objective, intercept, leaf quantile minimizer and optimizer settings, but use negative gradient q for strictly positive residual and q−1 for nonpositive residual. At residual zero, this chooses another mathematically valid endpoint of the pinball subdifferential; it is a numerical/optimization control, not a new statistical method.

All use 150 trees, learning rate .05, depth 2, leaf minimum 15, seed 20261028 and equal-region total weights. Fit in two predetermined orders: canonical profile order and its reverse. Do not choose an order using target performance. Report each order and the range; no best-order oracle as a method. V145 canonical `library_default` endpoints must reproduce archived predictions exactly. Calibration remains separate per threshold as in V145, with max month error within each calibration region. Preserve V146/V147 joint-calibration references with their different scope labelled.

## Checks and reporting

The strict-tie helper uses installed sklearn internals and is pinned to 1.8.0; fail for a different version. Check gradient values on positive/negative/zero residuals and preserve sample-weight/output-buffer semantics. Save training-order digests, first-tree node counts and impurity, initialization, source gradients, source calibration radii, target endpoints and input/code hashes. Source-response features must contain only the same legal monthly and proper-source information. Assert complete denominators, physical subset and alpha nesting; separately report lower/upper endpoint coverage.

Results can establish a stronger and more reproducible baseline. They cannot establish original methodology, hourly forecast coverage, superiority from a single favorable order, or an independent target-region validation. Reassess earlier CQR comparisons after this sensitivity check rather than overwriting archived output files.

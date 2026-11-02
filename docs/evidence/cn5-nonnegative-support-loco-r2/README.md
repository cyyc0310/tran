# Strict CN5 nonnegative-support interval diagnostic

Retrospective paired diagnostic on the already-scored 2024 strict five-group nine-country LOCO predictions. Original predictions and radii were hash-verified before scoring. No model was refit and no blind target was opened.

- Point: `max(0, p)`.
- Interval: `[max(0, p-r), max(0, p+r)]`, using each country/method source-calibrated 90% radius.
- Outputs: nine target prediction pairs; monthly, country-macro, and nine-country-macro metrics.
- Scope: fixed-factor generation-side proxy, nonnegative by construction; not measured CIF and not independent geographic confirmation.
- Result: coverage is exactly unchanged for all nine countries and methods; all point predictions were already nonnegative; interval widths and Winkler scores decrease or stay equal. Undercoverage remains.

See [diagnostic report](../../NONNEGATIVE_SUPPORT_CN5_LOCO_RESULTS_20261101.md) and `audit_manifest.json` for hashes.

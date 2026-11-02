# US↔GB dual-track nonnegative-support diagnostic

Retrospective paired diagnostic on the already-scored 2024 dual-track US↔GB replay. All 32 source prediction hashes and 16 target-response group hashes were checked before rescoring.

- Tracks remain separate: fixed-factor generation proxy and official EIA/NESO consumption-intensity estimates.
- Both monthly input arms remain separate in `region_macro_metrics.csv`; endpoint and directional summaries average them equally after monthly scoring.
- Point: `max(0, calibrated point)`.
- 90% interval: intersect the original source-calibrated interval with `[0, +infinity)`.
- Target labels in this observed 2024 cohort are all nonnegative. Empirical coverage is unchanged; the support transformation does not provide a future coverage guarantee.
- Gaussian CRPS is omitted for transformed forecasts because set intersection does not specify a full truncated predictive distribution.
- This target was already seen in development; it is not blind geographic confirmation.

See [results report](../../NONNEGATIVE_SUPPORT_DUAL_TRACK_RESULTS_20261101.md) and `audit_manifest.json`.

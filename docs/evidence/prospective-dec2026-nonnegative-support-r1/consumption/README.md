# US↔GB December 2026 nonnegative-support sensitivity

Derived from the sealed December 2026 consumption-side predictions at the same 2026-11-01 forecast origin. No target response was read.

- Ten files: seven main US BA forecasts, one GB forecast, and separate ISNE/MISO zero-clipped generation-mix sensitivities.
- Every file contains 744 rows.
- Calibrated point values and source-calibrated 90% interval endpoints are intersected with `[0, +infinity)` by applying `max(0, x)`.
- These are secondary sensitivity outputs. The primary sealed forecasts are unchanged; this is not a coverage guarantee or a selected winner.
- SHA-256 hashes and derivation status are in `evidence_manifest.json`, `prediction_seal.json`, and `status.json`.

See [pre-score audit](../../../PROSPECTIVE_DEC2026_PRE_SCORE_QA_20261101.md) and [primary protocol](../../../PROSPECTIVE_CONSUMPTION_US_GB_DEC2026_PROTOCOL.md).

# GB→US November 2026 prospective forecast evidence

Status: `SEALED_PROSPECTIVE_GB_TO_US_EIA923_NOV2026_FORECAST`; not scored. The 2026-11 target response must remain unopened until the target month has ended and an independent response is available.

The seven gzip CSV files contain 5,046 local-hour predictions for BPAT, CISO, ERCO, FPL, ISNE, MISO, and NYIS. They preserve method-level raw and source-calibrated predictions, as well as 5th/95th source-Gaussian reference bounds. The bounds are not calibrated to target coverage.

Inputs: official EIA-923 August 2026 release (published 2026-10-23), plant-month BA mapping through Page 6, finite reported fuel rows only, and each BA's local calendar. Minimum numeric row fraction was fixed at 90%. Source fitting uses GB 2022 and source calibration uses GB 2023. M+C is a separate 2020 coal/gas capacity reference arm. This is a known-region prospective time holdout, not an unseen-region test.

`provenance.json` records hashes for code, source data, and output files; all 50 entries were independently verified after the run. `prediction_seal.json` records the per-region forecast hashes. `preseal_target_isolation.json` records that no 2026 target-response files were present or read.

**Current archive audit correction (2026-11-01):** all 29 data hashes and 13 output hashes still match; 6 of 8 code hashes match. The originally hashed runner script is absent from the current worktree, and the originally hashed protocol is absent; the same-name protocol now available in `docs/` is a reconstruction and does not match the archived protocol hash. Thus 48/50 archived hashes are currently verifiable. The two-BA generation-proxy extension uses distinct `strict` script/protocol filenames and is not a replacement for the archived seven-BA consumption-side forecast.

Data note: the Page 1 August net-generation aggregates include negative hydro for ISNE and MISO from net storage/pumped-storage consumption. The input therefore represents a signed reported-subset net-generation composition. Preserve the sealed predictions as the primary result and report a zero-clipped/renormalized input sensitivity alongside scoring.

Main protocol: [`docs/PROSPECTIVE_GB_TO_US_EIA923_NOV2026_PROTOCOL.md`](../../PROSPECTIVE_GB_TO_US_EIA923_NOV2026_PROTOCOL.md). Status/report: [`docs/PROSPECTIVE_GB_TO_US_NOV2026_FORECAST_RESULTS.md`](../../PROSPECTIVE_GB_TO_US_NOV2026_FORECAST_RESULTS.md).

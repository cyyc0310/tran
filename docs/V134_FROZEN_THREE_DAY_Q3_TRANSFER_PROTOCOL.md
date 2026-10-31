# V134 — Frozen Three-Day Contract on the 2026Q3 Region Holdout

## Question

Does V132's selected three-day threshold-side information contract retain its interval-width gain on a 2026Q3 cohort of five regions outside V129's 21-region development cohort?

## Frozen design

- Freeze the three-day UTC block rule and the 3-field contract from V132: block thermal total, threshold high-side MWh, and clean total. No parameter is reselected using Q3 outcomes.
- Use V113's 5-region × 3-month (2026Q3) availability roster, retain the exact rows passing its frozen ≥80% hourly completeness screen, and report every exclusion.
- Compare monthly risk-side information, three-day risk-side information, and three-day detailed-fuel oracle bounds without technology capacity. Use the same V129 EF support intervals, τ=600/900, and ≥1/5/10% decision targets.
- Assert containment of nominal fixed-factor Q3 K in each risk-side interval and containment of the binned detailed robust envelope. Compare 0-error three-day endpoints with recomputation from the same source file; no outcome-based tuning.
- Record the V113 hourly source file hash and cohort, factor support, protocol/script hashes, and all assertion failures.

## Confirmation status

The 2026Q3 rows have previously been opened for other TransCIF experiments. They are spatially outside V129 and time-separated from the V129 development cohort, but this is a post-hoc method-transfer probe, not a blind confirmatory test. A future time block or newly obtained region must be reserved for a true confirmatory evaluation.

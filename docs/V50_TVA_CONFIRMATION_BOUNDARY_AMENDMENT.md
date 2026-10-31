# TVA label-boundary amendment (before corrected scoring)

Registered 2026-10-27, before corrected target metric computation. Parent protocol: [V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md](V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md).

## Failed first attempt

The first score command stopped before producing a metric or score file because it read only V6 native snapshot 8 (2026 Jan–Jun) and therefore constructed 180 rather than the frozen 181 UTC days. No interval was compared with a target value and no summary statistic was written. Target records were computationally opened during the failed label-construction path; this access is recorded and the data are not described as still technically unopened. The V6 metadata audit had already specified the correct fixed snapshot membership for the target semester: snapshots 7 and 8.

## Fixed boundary handling

The corrected label reader includes TVA rows from exactly V6 native snapshots 7 (2025 Jul–Dec) and 8 (2026 Jan–Jun), filtered only by valid_at in [2026-01-01T00:00Z, 2026-07-01T00:00Z). V6 audit maps the first hour of 2026 to snapshot 7 and all subsequent TVA hours to snapshot 8. Both files and schemas are hash-checked against the original V6 audit. Duplicate region/hour rows are rejected. The V6 full-day proxy quality rule is applied without interpolation or score-based deletion. The scored key set must exactly equal the preselected 181 UTC days. No target outcomes were inspected from the failed attempt; no candidate, comparison, threshold, period, or target rule changes here.

Corrected score code is frozen as scripts/score_v50_tva_asymmetric.py.

Scorer SHA256: 5f749b7ff2405aad97c221ab3b81305d5928531dd9a8dc393bd9262201d5a0f5

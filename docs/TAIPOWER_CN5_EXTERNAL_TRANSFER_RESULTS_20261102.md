# Taiwan CN5 external transfer: result and scope failure

Date: 2026-11-02  
Protocol: [`TAIPOWER_CN5_EXTERNAL_TRANSFER_PROTOCOL_20261102.md`](TAIPOWER_CN5_EXTERNAL_TRANSFER_PROTOCOL_20261102.md)  
Status: response opened after prediction seal; **quality gates failed, no model accuracy metrics are admissible**.

## Frozen prediction and response integrity

- Prediction file SHA-256: `7d767b19df85cd5df213b7bfcf6494b5f0c31ed2e5dba466a05c65ba4f10ddd6`.
- Frozen protocol SHA-256: `21a880c05c3ec71b5c1ec36fe63ec959d83f578f269a14b6abe6adb2d600896f`.
- The prediction covers 2,184 Q2 hours for each of three source configurations (US, GB, US+GB), with 6,552 rows total.
- The target response was retrieved after sealing. The official file returned HTTP 200, 194,124,659 uncompressed bytes, and SHA-256 `9bb758b380e0736c40ad01ff25435b79f63a59fe2d854a8c519928381a3f3020`.
- A small target sample had been read during preflight, so this is not described as a blind test.

## Why the scoring gate failed

| Gate | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Mapped generation support | 92.62% | ≥99% | Fail |
| April valid hours | 705/720 (97.92%) | ≥99% each month | Fail |
| May valid hours | 735/744 (98.79%) | ≥99% each month | Fail |
| June valid hours | 712/720 (98.89%) | ≥99% each month | Fail |
| Hourly mapped total vs nationwide monthly total | 47.0%, 55.8%, 54.1% relative gap | ≤10% | Fail |

The unclassified positive source is `汽電共生` (cogeneration), equivalent to about 2.79 TWh over the quarter when the 10-minute MW readings are integrated. It was kept unknown because the public record does not expose its fuel subtype; assigning it a thermal factor after seeing the response would violate the frozen mapping. The mapped hourly data also contain negative fuel readings. Their affected hours were excluded under the protocol, leaving monthly support below the required threshold.

No MAE, CRPS, coverage, interval width, or model ranking is reported. The saved `monthly_metrics.csv` and `summary_metrics.csv` are intentionally empty. The seal and gate details are in [`results/taipower-cn5-external-transfer-r1/scored/status.json`](../results/taipower-cn5-external-transfer-r1/scored/status.json) and [`score_seal.json`](../results/taipower-cn5-external-transfer-r1/scored/score_seal.json).

## Scope diagnosis

The hour-level label is much closer in scale to Taipower's own monthly table than to the nationwide total, but it still does not fully match Taipower's scope. Hourly mapped totals were 12.25, 11.06, and 11.65 TWh for April–June. Taipower's monthly totals were 14.73, 14.98, and 15.68 TWh; nationwide totals were 23.11, 25.03, and 25.39 TWh. Thus the observed-to-Taipower total ratios were about 83.1%, 73.8%, and 74.3%, while the nationwide comparisons were substantially worse.

This pattern suggests that the unit-level hourly file does not represent the complete nationwide generation mix and may be closer to Taipower-owned generation. It is still not a validated scope match: thermal, hydro, wind, and solar components differ, and the target monthly inputs used in the frozen prediction were nationwide. The public unit data therefore cannot support a valid external accuracy claim under this contract.

## Decision

Keep this as a documented external data/scope failure, not a negative model result. Do not relax the 99% support or 10% scope rules and do not reclassify cogeneration based on the outcome. A future Taiwan experiment needs a single consistent boundary for lagged monthly inputs and hourly labels, plus fuel-resolved and sufficiently complete public hourly response data. With the current public-only access, return to the primary cross-country benchmark and prioritize targets whose monthly input table and hourly generation labels demonstrably share the same system boundary before sealing predictions.

## Reproduction artifacts

- Frozen predictions: [`sealed_predictions.csv.gz`](../results/taipower-cn5-external-transfer-r1/sealed_predictions.csv.gz)
- Prediction seal: [`prediction_seal.json`](../results/taipower-cn5-external-transfer-r1/prediction_seal.json)
- Post-seal raw response snapshot: [`target_generation_10min_snapshot.json.gz`](../data/taipower-public-hourly-target-r1/target_generation_10min_snapshot.json.gz)
- Post-seal labels and audits: [`scored/`](../results/taipower-cn5-external-transfer-r1/scored/)
- Scorer: [`score_taipower_cn5_external_transfer.py`](../scripts/score_taipower_cn5_external_transfer.py)

# V395–V400: lagged mix transfer, external failure, and monthly-signal ablations

Date: 2026-10-31. These experiments use the EIA generation-derived fixed-factor CIF proxy, not direct measured emissions. They are not China hourly-CIF validation.

## Shanghai export handoff

The three user-supplied desktop CSVs are byte-identical to the archived Shanghai exports in `data/nbs_user_exports/shanghai/`: thermal SHA256 `92e02b10dbf5d19f3a680b481e005e5b6ada8a32c97d0bc7f805076cb5b10228`, solar `85941ab60c7a99bdeab88d43ca3c386cc409e5fe3a018640900a4abf3fdd8407`, and wind `7a6acef4e1503f7b80b7350adaaaea12a7bfe0fec639467c9532b99bdca11ad9`. V385 already parsed these files and matched all 33 nonempty current-month observations to the official panel to displayed precision. They cover 2025-09–12 and 2026-03–09; 2026-02 current value is blank, and no cumulative value was differenced into a monthly observation. Hydro and nuclear remain zero only where the user supplied that domain fact and the official series has a reported observation; missing months remain missing. Thus these files add provenance confirmation, not new historical coverage or a new experiment.

## Evidence from the transfer experiments

### V395: local lag-2 thermal share helps inside the source domain

Seven source areas, train 2022, calibrate 2023, test 2024, with leave-one-region-out fitting. At nominal 95%, the local lag-2 thermal-share Ridge reduced region-equal hourly MAE from 105.25 to 72.09 g/kWh and mean interval width from 530.67 to 435.45; interval score fell from 558.32 to 449.93. The paired region-cluster bootstrap favored local mix for width (gain 100.91, 95% interval 53.42–166.45) and interval score (130.10, 76.41–215.68), with all seven source regions positive. Tail-count coverage at 300 g/kWh was only 91.7% on average and 41.7% in the worst region. This was a development holdout, not external confirmation.

### V396–V397: conditional radii and direct tail frequency did not fix risk coverage

Conditioning calibration blocks on local mix narrowed hourly intervals modestly; the best checked bandwidth gave 98.6% aggregate hourly coverage, 93.6% worst-region coverage, width 415.95, and interval score 433.26. Tail-count coverage fell to 86.9%, with 50% worst-region coverage. Directly predicting monthly tail frequency also under-covered: for the 300 g/kWh threshold its point absolute fraction error was 0.290, but 95% count-interval coverage was 85.7%, worst region 0%. Neither branch meets the intended joint coverage/efficiency criterion.

### V398: the frozen source gain does not spatially transfer

The frozen V395 model was predicted for six external EIA areas over 2026H1, sealed before target-hour scoring. This is a frozen-method confirmation for this model, although those areas had appeared in other project development work. At nominal 95%, local mix reduced average width (559.44 to 502.40) and MAE (209.81 to 154.95), but coverage was only 77.8%, with worst-area coverage 12.2%. Paired six-area bootstrap intervals crossed zero for both interval-score and coverage gains. Threshold-count coverage was 61.1% at 500 g/kWh and 55.6% at 700 g/kWh, each with a worst-area value of zero. Therefore the source-domain result is not a transport breakthrough.

The already-opened external feature audit sharpens the failure diagnosis: all five eligible LGEE lagged shares are 0.980–0.990, above the maximum 0.761 among any source-area 2022 monthly shares; LGEE hourly coverage was 12.2%. PACE shares (0.461–0.624) lie inside the pooled source range, yet hourly coverage was still 72.1%. This supports two different failure modes: severe feature-support extrapolation for LGEE, and unresolved conditional shape/risk for PACE even without obvious scalar-support extrapolation. This is diagnostic evidence only; do not tune and rescore these six regions.

An additional V395 source-only diagnostic shows why a scalar range check is not itself the solution. Under each held-out fold, 17/71 eligible 2024 region-months fell outside the pooled 2022 donor-share min–max range. Those months had 99.9% hourly coverage and mean score 315.3, compared with 98.2% coverage and score 484.4 inside the range; all 12 BPAT months were below donor support but remained well covered. Conversely MISO had 9/12 months inside the pooled range yet only 92.2% mean coverage. The high-share LGEE case is an actionable extrapolation alarm, but a two-sided min–max abstention rule would reject benign low-share months and still miss in-support risk failures. This is a descriptive diagnostic on an already-used development set, not a calibrated OOD detector.

### V399: more composition detail is not a reliable substitute for risk information

In the same seven-source-area 2022/2023/2024 development setup, adding eight t-2 fuel shares and hour/month interactions reduced 95% hourly width and interval score relative to no-mix calendar Ridge (401.92 vs 530.67; 419.14 vs 558.32). However, compared with the simpler thermal-share arm it had worse MAE (80.79 vs 72.09), lower hourly coverage (97.0% vs 98.8%), lower 300-threshold count coverage (78.0% vs 91.7%), and zero worst-region count coverage. Its paired bootstrap contrast against the simpler thermal-share arm did not establish an hourly-width or interval-score gain (95% intervals crossed zero); it did not show a robust risk improvement. This eight-fuel feature set is also more information than the Chinese monthly statistics provide, where thermal generation is aggregated.

### V400 exploratory: monthly share change adds no tail-risk breakthrough

One predeclared public-information candidate was the change in local thermal share between the two most recently available monthly observations (`t-2` minus `t-3`). A direct monthly tail-fraction Ridge was fit on 2022 donor region-months, calibrated on 2023 donor region-months, and scored on seven leave-one-region-out 2024 targets (70 region-months per threshold). At 300 g/kWh, adding the share change to season plus share worsened mean absolute fraction error (0.3065 to 0.3164), interval score (1.6685 to 1.7143), and interval width (0.8254 to 0.8486); coverage moved from 82.9% to 84.3%, still far below nominal 95%. At 500 g/kWh, fraction error improved slightly (0.0456 to 0.0443) and width narrowed (0.1696 to 0.1611), but interval score worsened (0.7238 to 0.7285) and coverage fell (84.3% to 82.9%). Seven region clusters and one development year make this descriptive only. The simple monthly change does not justify another method branch.

## Research decision

No algorithmic breakthrough has been demonstrated. The defensible signal is now narrower: released lagged local composition can improve the transferred **level** in some source regions, but neither richer composition nor composition-conditioned residual calibration identifies local hourly tail risk. The external failures indicate that a useful framework needs an explicit support/abstention layer for level extrapolation and a separately validated local risk signal for temporal shape. A support warning can detect LGEE-like cases; it does not solve PACE-like cases.

For China, the user's monthly technology totals support a release-time-aware coarse-composition and threshold-identification benchmark, but not a claim of calibrated hourly CIF prediction. The next defensible algorithm test should use source-only splits to compare: (1) level transfer with a frozen support rule; (2) residual-risk transfer conditioned on one predeclared, legally available local temporal statistic; and (3) an honest partial-identification/abstention baseline. The statistic's value must be measured under equal information budgets and a new frozen target. Monthly generation mix alone is not evidence that hourly shape or tail risk is known. No new external target should be opened until such a rule wins on development data against simple anchoring and strong conditional baselines.

## Reproduction outputs

- V395: `results/v395-temporal-heldout-lagged-mix-r1/`
- V396: `results/v396-local-mix-conditioned-risk-r1/`
- V397: `results/v397-direct-tail-frequency-r1/`
- V398: `docs/V398_EXTERNAL_LAGGED_MIX_CONFIRMATION_PROTOCOL.md` and `results/v398-external-lagged-mix-r1/`
- V399: `results/v399-fuel-disaggregation-lag2-r1/`
- Shanghai identification: [V385 results](V385_SHANGHAI_RECENT_SHARP_K_RESULTS.md)

V400 was an inline exploratory calculation using the V369 EIA source panel and the V395 monthly feature construction; unlike V395–V399 it has no standalone runner, frozen artifact directory, or independent audit. Treat its numbers as a screening result only.

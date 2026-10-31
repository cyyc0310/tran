# V370 results: threshold-gated selective tail-count intervals

Protocol: [V370](V370_THRESHOLD_GATE_SELECTIVE_PROTOCOL.md). Runner: `scripts/v370_threshold_gate_selective_intervals.py`. Results: `results/v370-threshold-gate-selective-r1/`.

## Results

The threshold-side gate was tested as a prediction policy: return `[0,H]` on EF-support-crossing cases, or abstain; on non-crossing cases, calibrate asymmetric count residuals using only non-crossing training-region-months.

At τ=300, the gate crossed in roughly 83% of the 245 region-months. Returning `[0,H]` there and applying the conditional residual interval elsewhere gave 99.2% overall region-equal coverage at nominal 90%, but the average width was 83.3% of the month, wider than the always-issued local global asymmetric interval (79.3% width, 100% empirical coverage in this sample). Marking crossing cases as abstentions retained only 16.7% of region-months; conditional coverage among issued cases was **75%**, with one held-out region's sole issued month uncovered. Thus the conditional calibration failed its gate.

At τ=500, the gate retained 37.6%. Conditional intervals on retained cases were zero-width and covered all held-out counts; this occurs because those retained cases had no proxy tail events and is not evidence of general risk prediction. Assigning full-width sets to crossing cases yielded 100% coverage but 62.4% mean monthly width, versus narrower always-issued intervals that undercovered. At τ=700, all methods were effectively scoring a zero-event target.

## Decision

The analytic gate is useful as an **information warning**: the EF support and local thermal share identify a large subset where a narrow count estimate is unjustified. The gate-plus-abstention policy did **not** improve the coverage/width tradeoff under this dataset and calibration budget. At τ=300, non-crossing calibration was sparse and region-specific; when it was forced to issue, its coverage fell to 75%. Do not present V370 as a successful algorithm.

All results remain cross-validated within seven previously inspected U.S. balancing authorities against a fixed-factor generation proxy. They do not establish China transfer, measured emissions accuracy, or formal coverage guarantees.

# Joint Train Sanity: QLD1 seed 0

- Target: `QLD1`
- Sources: `NSW1, VIC1, SA1`
- Elapsed: `44.0s` (0.73 min)
- Origins: 12

## Stage 1 (ZS+ attention + BasisMix head)

- Initial train loss: `54.9436`
- Final train loss:   `50.4678`
- Final val MAE:      `27.7552`
- Steps: 30

## Stage 2 (+ per-direction correction)

- Initial train loss: `50.4253`
- Final train loss:   `50.1236`
- Final val MAE:      `27.1773`
- Steps: 30

## Verdict

- **Stage 2 val MAE 27.18 < 41 target**: GO
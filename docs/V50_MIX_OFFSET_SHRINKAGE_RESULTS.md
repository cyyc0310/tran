# V50 monthly-mix correction: fixed shrinkage path

Run date: 2026-10-27. The fixed grid and interpretation limits were registered in [V50_MIX_OFFSET_SHRINKAGE_PROTOCOL.md](V50_MIX_OFFSET_SHRINKAGE_PROTOCOL.md). No coefficient was selected for confirmation from this table.

## Source-macro path

| Offset multiplier | Hourly CRPS | Daily level MAE | 90% coverage | Width | Interval score | Particle clipping |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 (raw anchor) | 125.14 | 144.61 | 88.50% | 527.98 | 712.16 | 5.66% |
| 0.25 | 102.76 | 119.41 | 92.30% | 522.60 | 625.73 | 6.66% |
| 0.50 | 85.42 | 102.51 | 95.16% | 512.84 | 567.46 | 8.02% |
| 0.75 | 74.58 | 92.21 | 96.82% | 497.69 | 526.64 | 8.97% |
| 1.00 | 71.43 | 93.47 | 95.27% | 478.64 | 510.42 | 10.05% |

The unshrunk result has the best pooled proper scores, but the 0.75 point met the descriptive development gate on macro and worst-source coverage (worst source NWMT 86.64%; WACM 89.10%), with narrower width than the raw anchor. This is not confirmatory because the coefficient was inspected on the same ten previously used sources. Performance also varies sharply: at λ=0.75, SEC coverage is 97.62%, while NWMT and WACM are the limiting sources. At λ=1.00, SEC collapses to 75.18%.

SCL remains problematic across the curve: clipping rises from 56.46% in the raw anchor to 66.5%, 79.6%, and 86.1% at λ=.25/.50/.75. Its low CRPS and narrow intervals under the corrected paths cannot be treated as healthy calibration when most particle-hours are outside the declared physical support before clipping.

## Decision

Uniform shrinkage exposes a bias/coverage tradeoff, but no coefficient resolves all source failures cleanly. λ=.75 is a candidate worth stress-testing, not a frozen method: clipping at SCL remains severe, and its apparent source gate is post-selected on development data. The next support-guard test is recorded separately. No unseen source has been opened.

## Artifacts

- Script: `scripts/probe_v50_mix_offset_shrinkage.py`
- Output: `results/disjoint-calibration-v50-mix-offset-shrinkage/`
- Run summary records frozen protocol/input/output hashes.

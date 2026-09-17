# LOJO Verdict: Leave-One-Jurisdiction-Out on the FD-41 stack

**Date**: 2026-09-17 ｜ **Task**: paper-gap #3 — re-run the paper's
strongest claim (US+UK-trained → predict all 4 AU grids) on the current
fuel-decomposition stack, replacing stale fused-five-era numbers.

**Protocol**: identical to official FD-41 (`fuel_decomp_eval_full_fd41.json`):
epochs 900, p_cold 0.3, use_monthly, dynamic_residual, wind_route_tau 1.1
(global), max_windows 700, seeds 0-4.  New flag `--cross-jurisdiction-sources`
excludes ALL regions in the target's jurisdiction from training sources
(4 AU targets → sources = 8 US + 17 UK regions).

## Per-region results (5 seeds)

| target | I_0 median | I_cfg median | I_+ median | I_+ 5-seed ens | persistence | PatchTST-sup | I_+ vs persist | I_+ vs sup |
|---|---|---|---|---|---|---|---|---|
| QLD1 | 30.4 | 33.7 | **19.1** | 19.1 | 20.2 | 30.3 | WIN −5.5% | **WIN −37%** |
| NSW1 | 53.4 | 60.3 | **42.3** | 42.3 | 48.0 | 46.8 | WIN −11.9% | **WIN −10%** |
| VIC1 | 88.1 | 87.0 | **81.3** | 81.3 | 98.6 | 92.3 | WIN −17.5% | **WIN −12%** |
| SA1 | 71.5 | 75.5 | 69.3 | 69.3 | 77.6 | 50.6 | WIN −10.7% | lose +37% |

(PatchTST-sup = 2023-protocol supervised numbers from the paper, unchanged —
supervised baselines are independent of our stack.)

## Comparison to the paper's stale claim

| metric | old paper (fused-five era) | new FD-41 stack |
|---|---|---|
| QLD1 ZS+ MAE | 27.0 | **19.1** |
| AU regions beating supervised PatchTST | 1/4 (QLD1) | **3/4** (QLD1/NSW1/VIC1) |
| AU regions beating persistence | 4/4 | 4/4 |
| median persistence improvement | ~12% | ~11.4% (range 5.5-17.5%) |

**The LOJO headline UPGRADES**: the claim survives the stack change and
strengthens — zero-shot cross-continent transfer now beats *supervised*
models on 3 of 4 AU grids (old: 1 of 4).  SA1 remains the honest exception
(classifier-synthetic fuel labels; PatchTST learns the label quirks from
10 months of local data, our zero-telemetry path cannot).

## Notes

- I_+ ensemble = seed-mean gives no extra gain (19.1→19.1 etc): ZS+
  calibration is level-anchored to target telemetry, which already
  absorbs seed variance; only I_0/I_cfg benefit from ensembling (e.g.
  QLD1 i0 33.7→30.8).
- I_0 (telemetry tier) LOJO is *worse* than persistence on NSW1/VIC1/SA1
  (53.4 vs 48.0; 88.1 vs 98.6 wins; 71.5 vs 77.6 wins) — only the ZS+
  calibration tier consistently wins.  Same pattern as LORO: transfer
  value concentrates in I_+.
- Cross-jurisdiction seed stability is high: I_+ spread across 5 seeds
  ≤ 0.1 for every region (42.3/81.3/69.3/19.1 stable to rounding).

## Repro

```bash
PYTHONPATH=src .venv-nemed/bin/python scripts/experiments/run_fuel_decomp_eval.py \
  --regions QLD1 NSW1 VIC1 SA1 --seeds 0 1 2 3 4 --epochs 900 \
  --monthly-config --dynamic-residual --cross-jurisdiction-sources \
  --out results/lojo_au_full.json
PYTHONPATH=src .venv-nemed/bin/python scripts/experiments/lojo_ensemble.py
```

Artifacts: `results/lojo_au_full.json`, `results/lojo_ensemble.json`,
`results/lojo_au_full.log`.  Code: `--cross-jurisdiction-sources` in
`run_fuel_decomp_eval.py`, `cross_jurisdiction` param in
`src/transcif/models/zeroshot/fuel.py` (`train_fuel_zero_shot`).

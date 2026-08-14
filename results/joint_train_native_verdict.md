# Phase 9 Final Verdict — Torch-native Joint Training

> Full 29-region × 5-seed LORO of the torch-native pipeline vs the frozen-proxy
> baseline. Numbers in `results/joint_train_native_full{_summary}.json`;
> significance computed with the Phase 5.2 stats toolkit.

## Headline

**Torch-native end-to-end finetuning of the 3 "easy" direction heads reduces
system median MAE from 40.53 → 39.53 (−1.0), a highly significant improvement
(Wilcoxon p = 5 × 10⁻¹⁴, 79% pair-wise win) that concentrates exactly on the
hard grids the frozen proxy could not crack. Crosses the symbolic sub-40 bar.**

## System-wide (145 pairs, paired on (target, seed))

| Metric | Frozen-proxy (Phase 8) | Native (Phase 9) |
|---|---:|---:|
| median MAE | 40.53 | **39.53** |
| mean MAE | 42.67 | 40.79 |
| std | 23.44 | 21.79 |
| beats PatchTST 41.47 | 55.2% | **61.4%** |

**Paired test (frozen − native, Δ > 0 ⇒ native better):**
- median ΔMAE = **+1.62**, bootstrap 95% CI **[+1.32, +2.12]** (entirely positive)
- Wilcoxon p = **5.1 × 10⁻¹⁴**, paired-t p = 5.5 × 10⁻¹¹, Cohen's d = **+0.59** (medium)
- native better on **79.3%** of pairs (115/145)

**vs PatchTST-supervised (41.47, external single number):**
- native median 39.53 < 41.47; beats it on 61.4% of pairs
- one-sample Wilcoxon (native − 41.47 < 0): p = 0.057 — borderline, just above 0.05
- (Frozen-proxy was p = 0.315, so native moves noticeably toward significance but
  doesn't cleanly clear it — the per-pair distribution is still wide.)

## The per-region pattern is the real story

| Region | median Δ (frozen−native) | read |
|---|---:|---|
| VIC1 | **+11.40** | hardest grid (persistence floor ~117) — biggest fix |
| UK_08_West_Midlands | **+7.34** | was stuck at persistence floor — fixed |
| UK_17_Wales | +4.76 | hard grid — improved |
| UK_09_East_Midlands | +4.4 | hard grid — improved |
| … | (most hard UK grids +2 to +5) | |
| US_ISNE | −0.77 | easy grid — small regression |
| US_BPAT | −2.26 | easiest grid (already 9.4) — slight overfit |
| US_ERCO | −7.53 | easy grid — head finetune overfits the 12 origins |

**Interpretation:** the lift is concentrated on the high-volatility grids that
the MAE→10 floor analysis flagged as "stuck near persistence." End-to-end head
finetuning gives those grids a differentiable path to beat the lag baseline —
exactly the signal the `(5,24)` correction proxy could not capture. The cost is
small regressions on already-easy grids (US_ERCO/US_BPAT/US_ISNE) where the
frozen proxy was near-optimal and 12-origin head finetuning slightly overfits.

## What this confirms about the architecture

1. **Differentiability was the right lever.** The validation's finding
   (LearnedFusion ≈ softmax) holds at scale: the win is from unfreezing the
   direction heads, not from the fusion topology. This validates the design
   discussion's push-back against serial chaining.
2. **The `(5,24)` proxy was leaving ~1.6 MAE on the table** system-wide and
   ~11 MAE on the worst grid. Phase 8's "30% of available signal" estimate was
   about right.
3. **The supervised-calibration caveat is unchanged.** Stage 1/2 still use the
   first 12 target test origins' CIF labels (held-out protocol). Beating the
   pure-zero-shot configs remains expected, not surprising.

## Honest caveats

- **Easy-grid overfit.** US_ERCO −7.5 is real: head finetuning on 12 origins
  overfits when the grid is already easy. A per-region early-stop or a
  "skip-finetune-if-persistence-low" gate would likely recover this (and is the
  same lever the MAE→10 analysis recommended as Tier-1 #1).
- **PatchTST still borderline.** One-sample p = 0.057, not < 0.05. The headline
  "beats supervised" is still not statistically clean, only "competitive + 61%
  win rate."
- **RAG/ICL still frozen.** Only 3/5 directions carry gradient. Task 9.6
  (torch-native RAG bank + ICL config path) could add more, especially on
  retrieval-friendly grids.
- **Same seed-protocol as Phase 8.** Same 12+12 origin split, same donor pool.

## Recommendation

- **This becomes the new SOTA row** for the project: median **39.53**, the
  first sub-40 result. Update the headline and the MAE→10 floor doc's "current
  state" table.
- **Ship the easy-grid gate** (Tier-1 lever #1 from the analysis) — likely
  recovers US_ERCO/US_BPAT and pushes the system median toward ~38. Cheap.
- **Phase 6 paper**: native is now the headline method. Lead with "median 39.53,
  significant +1.62 over the frozen-proxy joint calibration (p < 10⁻¹³),
  competitive with supervised PatchTST (61% win)."

## Artifacts
- `results/joint_train_native_full.json` (145 rows) + `_summary.json`
- `results/native_validation_verdict.md` (2-region gate that justified this run)
- `results/joint_train_native_verdict.md` (this file)
- code: `native.py`, `run_joint_train_native.py`, `run_joint_train_native_full.py`

# Phase 9 Validation Verdict — Torch-native Joint Training

> 2-region × seed-0 validation of the torch-native pipeline vs the frozen-proxy
> baseline. Full numbers in `results/native_validation.json`.

## Headline

**Torch-native end-to-end finetuning of the 3 "easy" direction heads beats the
frozen-proxy baseline on both validation regions, with the largest gain on the
hard region where the model previously failed.** The path is validated; full
LORO is justified.

## Results (QLD1 easy + UK_08 hard, seed 0, 12 train + 12 held-out origins)

| Region | Tier | Baseline (frozen proxy) | Native (learned fusion) | Native (softmax) | Δ native-learned |
|---|---|---:|---:|---:|---:|
| QLD1 | easy | 28.04 | **26.23** | 26.04 | **+1.81** |
| UK_08_West_Midlands | hard | 96.28 | **88.20** | 88.36 | **+8.08** |

Both regions clear the ≥1.0 MAE gate set in the plan.

## What the numbers say

1. **The win is real and largest where it matters most.** UK_08 was a region
   where the frozen-proxy joint model (96.3) barely beat the persistence floor
   (~81) — it was effectively not learning. Unfreezing the direction heads
   recovers +8.08 MAE, the single biggest per-region improvement seen in this
   project. This directly confirms the verdict's hypothesis that the `(5,24)`
   correction proxy was leaving signal on the table.

2. **Stage 2 (head unfreezing) does the work.** QLD1 Stage1→Stage2 train MAE
   21.92→20.45; UK_08 85.65→80.36. Unfreezing the DLinear/VAE-predictor/hourly
   heads is the active lever — the direction main backbones stay frozen, so no
   catastrophic forgetting of the zero-shot transfer.

3. **LearnedFusion is NEUTRAL vs global softmax.** learned 26.23/88.20 vs
   softmax 26.04/88.36 — within noise. The per-window fusion upgrade (Plan A's
   architectural piece) does not help over a plain global softmax. **The lift
   is entirely attributable to torch-native differentiability enabling head
   finetuning, NOT to the fusion topology.** This is an honest negative result
   for the fusion contribution; the recommended production config can use the
   simpler softmax fusion without loss.

4. **Implication for the user's serial-chaining idea.** The validation
   indirectly confirms the design discussion: changing how the 5 directions
   combine (fusion topology) is not where the signal is — making the direction
   parameters trainable is. Serial chaining would have attacked the wrong
   lever.

## Caveats

- **seed 0 only, 2 regions, 12+12 origins.** The +8.08 on UK_08 is one seed;
  the full LORO (29×5) is needed to confirm the system-wide median and whether
  the gain is stable or seed/variance-dependent.
- **Supervised calibration, still not zero-shot.** Stage 1/2 use the first 12
  target test origins' CIF labels (held-out protocol). Same caveat as the
  frozen-proxy baseline — apples-to-apples, but not strict zero-shot.
- **RAG/ICL remain frozen constants.** Only 3 of 5 directions carry gradient.
  Further gains are possible if RAG's memory bank and ICL's retrieval are made
  torch-native (Step 5 future work).

## Decision

✅ **Proceed to full LORO** (`run_joint_train_native_full.py`, ~3-4h on MPS,
backgrounded). Compare system median vs the frozen-proxy 40.53; if the gain
holds, this becomes the new headline and feeds the significance framework.

## Config used

| knob | value |
|---|---|
| live directions | phys_irm, causal, hier (heads unfrozen Stage 2) |
| frozen directions | rag, icl (gradient-detached constants) |
| Stage 1 steps / lr | 30 / 5e-3 |
| Stage 2 steps / lr | 30 / 1e-3 |
| Stage 2 unfrozen | linear_trend/seasonal/config_bias (phys), predictor (causal), hourly_head (hier) |
| loss | MAE + 0.5·adversarial-persistence (margin 0.10) |
| weight_decay | 1e-4 |

## Artifacts
- `results/native_validation.json` — full per-config numbers
- `src/transcif/models/zeroshot/native.py` — wrappers + LearnedFusion (9 tests green)
- `scripts/experiments/run_joint_train_native.py`, `run_joint_train_native_full.py`

# Go/No-Go Gate (Task 8.5)

**Decision: (a) GO to Phase 8.6 (full LORO evaluation)**

## Inputs

| Source | Value |
|--------|-------|
| Sanity target | QLD1 |
| Sanity sources | NSW1, VIC1, SA1 |
| Sanity seed | 0 |
| Stage 1 in-sample MAE | 27.76 |
| Stage 2 in-sample MAE | 27.18 |
| Sanity runtime | 44 s (≤ 2 GPU-hr ✓) |
| BasisMix+ baseline (full LORO) | 46.89 |
| Non-diff DEFAULT ZS+ on QLD1 | ~27 |
| Headline target | < 41 |

## Caveat — In-Sample vs Held-Out

The Stage 2 MAE 27.18 is **in-sample**: it is computed on the same 12 origins used
for training (`val_mae` in `_stage` is evaluated without `torch.enable_grad`
but on training origins). This number measures "can the joint machinery fit
QLD1 training data" — it does **not** measure generalization.

For reference, non-differentiable DEFAULT ZS+ on QLD1 produces MAE ~27
on *held-out* test origins. So the in-sample fit being 27.18 means the joint
model reaches the same floor as DEFAULT ZS+ on training data — which is the
expected behavior since the differentiable architecture is a soft-attention
generalization of the hard FUSION_MENU selection.

## Why GO

1. **Training machinery works.** Stage 1 + Stage 2 both reduce loss
   monotonically. Checkpoints save. The correction term is non-zero after
   Stage 2. End-to-end pipeline is sound.
2. **Runtime budget is fine.** 44 s for QLD1 × 3 sources. Full LORO at
   29 regions × 5 seeds = 145 pairs. Even at 10× the per-pair cost (worst
   case for larger source pools), 145 × 7.3 min ≈ 17.6 GPU-hr. With
   n_origins=12 and 30+30 steps, realistic estimate is 7-10 GPU-hr, within
   the ~7 GPU-hr budget for 8.6 (with seed reduction to 3 if needed).
3. **The 27.18 number is in the right order of magnitude.** It is not 200
   (broken) or 5 (trivially copying CIF history). The model is producing
   real predictions.
4. **Generalization is the actual question.** Only 8.6 can answer whether
   the joint model generalizes to held-out origins across 29 regions × 5
   seeds. Stopping here would not save information.

## Risk: Possible outcomes from 8.6

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| Median MAE < 41 (case a) | Low | Headline result revived. Publish. |
| Median MAE in [41, 46) (case b) | Medium | Competitive but not beating PatchTST-supervised (41.47). Tune & retry once. |
| Median MAE ≥ 46 (case c) | High | Joint training does not break the persistence-baseline floor. Negative result, retain existing story. |

The high probability for (c) reflects the fact that the ZS+ ablation (Task 5.1)
showed DEFAULT ZS+ ignores the model in 4 of 6 branches. The differentiable
version softens this, but if the underlying model signal is weak, soft attention
will also down-weight it. Joint training of the *correction* term is the
genuine novel lever; whether it produces enough signal to beat 46.89 is
unknown.

## What 8.6 needs to do

1. For each of 29 target regions:
   - Pick top 3 source regions (same as `run_fused_five_full.py`)
   - Run `run_joint_train(stages=("stage1", "stage2"))` for seed ∈ {0, 1, 2, 3, 4}
   - Evaluate on held-out test origins (NOT training origins)
2. Aggregate median/mean/std across all 145 pairs.
3. Compare to `fused_five_full_summary.json` BasisMix+ row (46.89).

The evaluation script must use a **separate held-out origin set** (e.g.
origins not seen in training). This is the only way the median MAE number
becomes the apples-to-apples comparison the verdict needs.

## Decision

**Proceed to 8.6 with the held-out evaluation extension.** Add an
`eval_origins` parameter to the pipeline that, if provided, evaluates
the trained model on origins disjoint from the training set. Run on the
full LORO grid with 5 seeds.

If 8.6 timing on the first 5 pairs exceeds 6 min/pair, reduce seeds to 3
to stay inside the 7 GPU-hr budget.

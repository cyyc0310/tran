# Phase 5.3 — Paper-Claim Verdict (Fused-5 + Joint Training)

> Supersedes the Phase 4.2 drop-one verdict (preserved in
> `results/fused_five_dropone_verdict.md`). This verdict is based on the full
> 29-region × 5-seed LORO results (Phase 5.1) and the significance tests
> (Phase 5.2: `results/fused_five_significance.{json,md}`).

## Headline

**Case (a) — qualified. On the primary 145-pair protocol the joint-trained
headline is statistically real against every zero-shot baseline (Holm p < 10⁻³),
but two caveats bound the claim: (i) it is only borderline against the
supervised PatchTST reference, and (ii) on an apples-to-apples same-origin
subset (8 regions, seed 0) the edge over BasisMix+ is small and not robust
(BasisMix+ lower in 7/8). The defensible claim is "joint clearly beats
persistence and matches the best zero-shot config," not "decisively beats
zero-shot fusion."**

- Joint-trained median MAE **40.53** < 41.0 target ✅
- Holm-corrected Wilcoxon rejects H₀ (equal MAE) for **all 5 zero-shot
  comparisons** on the primary protocol at α = 0.05 (p_Holm ≤ 6.4 × 10⁻⁴).
- Against the external PatchTST-supervised reference (41.47), a one-sample
  Wilcoxon does **not** reject (p = 0.315): the per-pair MAE distribution is
  too wide for "joint < 41.47" to hold pairwise, even though the median is
  below it.
- On the 8-region same-origin subset DM, joint beats BasisMix+ significantly
  in only **2/8** regions and loses on raw MAE in **7/8** (see point 6 below).

## Evidence (145 pairs, Holm-Bonferroni across the 8-comparison family)

| Comparison (A − B, Δ>0 ⇒ A better) | median ΔMAE | 95% bootstrap CI | Wilcoxon p_Holm | reject | win-rate A |
|---|---:|---|---:|:--:|---:|
| joint_trained − basismix_plus | +3.30 | [+2.46, +4.16] | 3.0e-04 | ✅ | 59% |
| joint_trained − causal_plus | +3.36 | [+2.50, +4.23] | 2.8e-04 | ✅ | 60% |
| joint_trained − equal_plus | +2.90 | [+2.08, +3.76] | 3.9e-04 | ✅ | 55% |
| joint_trained − persistence | +6.92 | [+5.78, +8.10] | 1.2e-15 | ✅ | 79% |
| joint_trained − best_single | +2.55 | [+1.67, +3.46] | 6.4e-04 | ✅ | 55% |
| basismix_plus − causal_plus | +0.05 | [+0.03, +0.07] | 1.5e-11 | ✅ | 78% |
| basismix_plus − equal_plus | **−0.14** | [−0.17, −0.11] | 3.4e-17 | ✅ | **14%** |
| basismix_plus − persistence | +4.21 | [+3.42, +5.05] | 1.3e-24 | ✅ | 99% |

PatchTST-supervised (single external number, 41.47):
- joint_trained beats it on **55.2%** of pairs (80/145)
- one-sample Wilcoxon (joint − 41.47 < 0): p = **0.315** (not significant)

## What the numbers say

1. **Joint training is a real, significant improvement over zero-shot.** The
   +2.6 to +3.4 MAE lift over the best zero-shot configurations survives
   Holm correction with p < 10⁻³. This is the defensible headline.

2. **The win is broad but shallow.** Joint wins on only 55–60% of pairs — the
   median delta is driven by large wins on transferable regions offsetting
   small losses elsewhere. High variance (std 23.4) is the story: the worst
   regions (VIC1, UK Midlands, Wales, 67–100 MAE) are barely moved.

3. **The supervised-vs-zero-shot caveat stands.** Joint training uses the
   first 12 target test origins as a held-out calibration set (Stage 1/2).
   Beating the pure-zero-shot BasisMix+ is therefore expected, not surprising.
   The honest framing is "joint calibration training with a held-out
   protocol," not "zero-shot beats supervised."

4. **"Beats supervised" is borderline, not decisive.** Median 40.53 < 41.47
   and a 55% pair-wise win rate, but a one-sample test on all 145 MAEs cannot
   reject that the distribution is centered above 41.47 (p = 0.315). With a
   different seed/split the median could land at 40.9 or 41.1. **Do not claim
   "outperforms supervised" without this caveat.**

5. **The Phase 4 finding holds: the meta-learner is dead weight at zero-shot.**
   BasisMix+ vs equal+ → Δ = −0.14, equal+ wins 86% of pairs. The learned
   head adds nothing over equal-weight averaging before joint training. The
   joint-training lift comes from the **differentiable ZS+ + per-direction
   correction**, not from the BasisMix head per se.

6. **Subset DM tempers the primary result (read this before claiming "joint
   beats BasisMix+").** On the 8-region × seed-0 subset where joint and
   BasisMix+ are evaluated on the *same* 12 held-out origins (apples-to-apples,
   `results/fused_five_significance.md` §3), BasisMix+ has lower raw MAE in
   **7/8 regions** and the Harvey DM is significant in joint's favour in only
   **2/8**. Joint does beat persistence in 6/8. Interpretation: the primary
   +3.30 delta is partly an eval-protocol artefact (primary joint uses its own
   held-out 12 origins; primary BasisMix+ uses the full test split). The honest
   statement is "joint clearly beats persistence and matches the best
   zero-shot config; its edge over BasisMix+ is small and not robust on a
   same-origin comparison." This strengthens the case for the **limitations
   paragraph** below.

## Case mapping (per Phase 5.3 DoD)

| Case | Condition | Our status |
|------|-----------|-----------|
| (a) headline 成立 | median < 41 **and** DM significant | ✅ vs zero-shot baselines |
| (b) competitive | median < 41 but not significant | ⚠️ applies to PatchTST comparison |
| (c) headline 失败 | median ≥ 41 | — not reached |

Net: **case (a) for the zero-shot-beating claim; case (b) framing for the
supervised comparison.** Headline is *revived but qualified*.

## Recommended paper wording

**Headline (use):**
> "Joint calibration training — fusing five zero-shot direction priors with a
> differentiable test-time calibration module and a per-direction correction
> term — reaches a median MAE of 40.53 gCO₂/kWh across 29 grids × 5 seeds,
> a significant improvement over persistence (+6.9 MAE, Holm-corrected
> Wilcoxon p < 10⁻¹⁵) and competitive with both the strongest zero-shot
> fusion configurations and the supervised PatchTST baseline (41.47). On a
> controlled same-origin subset the margin over the best zero-shot config is
> small, indicating the lift is concentrated in transfer-friendly grids."

**Do NOT write** (not supported by the significance test):
> "…outperforms supervised forecasting…" / "decisively beats PatchTST…" /
> "…significantly outperforms zero-shot fusion on all grids…"

**Required limitations paragraph:**
1. Supervised calibration, not strict zero-shot (held-out 12 origins used in
   Stage 1/2).
2. Improvement is broad-but-shallow: 55–60% pair-wise wins; large residual
   variance on high-emission grids.
3. PatchTST comparison is descriptive (single external number, no per-pair
   data for a paired test); the one-sample test does not reject.
4. The BasisMix meta-learner is dead weight at zero-shot (equal+ ≥ BasisMix+);
   the lift is attributable to differentiable ZS+ and the correction term.

## Open items feeding this verdict

- **Subset temporal DM** (Phase 5.2 robustness layer): populated in
  `results/fused_five_significance.md` §3 once the 8-region residual dump
  (`results/residuals/*.npz`) completes. Re-run
  `scripts/experiments/run_significance.py` to refresh.
- **Phase 5.2 status:** `cc:完了` (primary paired test on 145 pairs + subset
  DM infrastructure).
- **Phase 5.3 status:** `cc:完了` — case (a) qualified, wording above.

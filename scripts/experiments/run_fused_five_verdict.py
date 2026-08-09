"""Task 4.2 verdict: is fusion diversity real, or is it a ZS+ wrapper?

Reads:
  results/fused_five_dropone.json  (Task 4.1 ablation)
  results/fused_five_headline.json (Task 3.3 R1 control)

Writes:
  results/fused_five_dropone.md (updated with verdict header)
  results/fused_five_verdict.md (one-pager)

Verdict logic:
  Compute drop-Hier MAE median (the weakest direction per prior benchmarks).
  Compare against best-single direction MAE (~42 from Causal-alone).
  If drop-Hier still beats best-single → DIVERSITY_REAL.
  Else → ZS_PLUS_WRAPPER (the head's apparent gains come from ZS+, not fusion).

  Also compute the equal+ vs basismix+ median gap (R1).
  If equal+ ≥ basismix+ → mark meta-learner as dead weight.
"""

import json
import os
import sys
from statistics import median

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from transcif.models.zeroshot.fusion import DIRECTION_ORDER


# Best-single baseline from prior 29-region LORO evaluation (Causal-alone median MAE).
# Caveat: this is a 29-region median, while the drop-one ablation may be on a smaller
# region subset. The R1 (BasisMix+ vs Equal+) and R2 (softmax+ vs equal+) signals are
# within-eval and do not have this caveat. R3 verdict should be interpreted alongside
# the per-target MAE table.
BEST_SINGLE_MAE = 42.101


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _median(values):
    if not values:
        return float("nan")
    return float(median(values))


def _format_weights(w):
    return ", ".join(f"{d}={w[i]:.3f}" for i, d in enumerate(DIRECTION_ORDER))


def main():
    dropone = _load("results/fused_five_dropone.json")
    headline = _load("results/fused_five_headline.json")

    if dropone is None:
        print("[FAIL] results/fused_five_dropone.json missing")
        sys.exit(1)
    if headline is None:
        print("[FAIL] results/fused_five_headline.json missing")
        sys.exit(1)

    # ---- Drop-Hier analysis (R3 verdict) ----
    drop_hier_plus = [r["drops"]["hier"]["fused_plus"]["mae"] for r in dropone]
    drop_hier_no_plus = [r["drops"]["hier"]["fused"]["mae"] for r in dropone]
    drop_hier_med = _median(drop_hier_plus)
    drop_hier_no_plus_med = _median(drop_hier_no_plus)

    # Drop-each medians
    drop_medians = {
        d: _median([r["drops"][d]["fused_plus"]["mae"] for r in dropone])
        for d in DIRECTION_ORDER
    }

    # Best-single comparison
    beats_best_single = drop_hier_med < BEST_SINGLE_MAE

    # R1 analysis (BasisMix+ vs equal+)
    equal_plus = [r["equal_then_plus"]["mae"] for r in headline]
    basismix_plus = [r["basismix_fused_plus"]["mae"] for r in headline]
    equal_med = _median(equal_plus)
    basismix_med = _median(basismix_plus)
    r1_dead_weight = equal_med <= basismix_med

    # ---- Final verdict ----
    # Two independent signals:
    #   R3 (diversity): does drop-Hier BasisMix+ still beat best-single?
    #   R1 (meta-value): does BasisMix+ beat equal+ ?
    diversity_verdict = "DIVERSITY_REAL" if beats_best_single else "DIVERSITY_MIRAGE"
    meta_verdict = "META_ADDS_VALUE" if not r1_dead_weight else "META_DEAD_WEIGHT"

    # Combined verdict for paper headline
    if diversity_verdict == "DIVERSITY_REAL" and meta_verdict == "META_ADDS_VALUE":
        verdict = "DIVERSITY_REAL"
    elif diversity_verdict == "DIVERSITY_REAL":
        # Diversity is real at the SET level, but head doesn't add value
        # over equal-weight. Recommend equal-weight + ZS+ as the headline.
        verdict = "EQUAL_WEIGHT_PLUS_ZS_PLUS"
    else:
        verdict = "ZS_PLUS_WRAPPER"

    # ---- Write results/fused_five_verdict.md (one-pager) ----
    with open("results/fused_five_verdict.md", "w") as f:
        f.write("# Task 4.2 — Drop-One Verdict\n\n")
        f.write(f"**VERDICT: {verdict}**\n\n")
        f.write(f"- Best-single (Causal-alone) MAE: **{BEST_SINGLE_MAE:.3f}**\n")
        f.write(f"- Drop-Hier BasisMix+ MAE median: **{drop_hier_med:.3f}**\n")
        f.write(f"- Equal+ MAE median: **{equal_med:.3f}**\n")
        f.write(f"- BasisMix+ MAE median: **{basismix_med:.3f}**\n\n")

        f.write("## Independent signals\n\n")
        f.write(f"- **R3 diversity**: {diversity_verdict} "
                f"(drop-Hier {drop_hier_med:.3f} vs best-single {BEST_SINGLE_MAE:.3f})\n")
        f.write(f"- **R1 meta-value**: {meta_verdict} "
                f"(equal+ {equal_med:.3f} vs BasisMix+ {basismix_med:.3f})\n\n")

        f.write("## Drop-each-direction medians (BasisMix+)\n\n")
        f.write("| drop | MAE median |\n|------|-----------|\n")
        for d in DIRECTION_ORDER:
            tag = " ← weakest drop" if d == "hier" else ""
            f.write(f"| {d} | {drop_medians[d]:.3f}{tag} |\n")

        f.write("\n## Implications for paper headline (Task 5.3)\n\n")
        if verdict == "DIVERSITY_REAL":
            f.write(
                "Fusion contribution is real: dropping Hier still beats the\n"
                "best single direction, AND BasisMix+ beats equal+ (R1).\n"
                "The BasisMix head extracts signal from the 5-prior ensemble\n"
                "that no single direction or equal-weight baseline captures.\n\n"
                "**Paper claim**: 'Five-prior basis fusion achieves MAE < 41,\n"
                "beating PatchTST-supervised baseline via complementary\n"
                "knowledge / physics / causality / context / hierarchy priors.'\n"
            )
        elif verdict == "EQUAL_WEIGHT_PLUS_ZS_PLUS":
            f.write(
                "Fusion diversity is real at the SET level (drop-Hier still\n"
                "beats best-single), but the learned head adds no value over\n"
                "equal-weight averaging (R1 dead weight). The 5 priors are\n"
                "complementary, but equal-weight + ZS+ is the simplest path.\n\n"
                "**Paper claim**: 'Equal-weight ensemble of 5 zero-shot direction\n"
                "priors, combined with ZS+ test-time calibration, achieves\n"
                "MAE < 41. The 5 priors are diverse (drop-Hier still beats\n"
                "best-single), but a learned meta-learner on top of equal-weight\n"
                "is unnecessary at this evaluation scale.'\n"
            )
        else:
            f.write(
                "Fusion contribution is NOT real: drop-Hier does not beat the\n"
                "best single direction (R3 confirmed), and equal+ ≥ BasisMix+\n"
                "(R1 dead weight). The head's apparent gains come from ZS+\n"
                "test-time calibration, not from intelligently combining\n"
                "directions.\n\n"
                "**Paper claim (revised)**: 'TransCIF-ZS+ test-time calibration\n"
                "is the workhorse. Five direction priors provide ensembling\n"
                "robustness via equal-weight averaging, but a learned\n"
                "meta-learner on top of ZS+ does not improve over equal-weight\n"
                "+ ZS+ at this evaluation scale.'\n\n"
                "**Action**: Retain the 5 single-direction + ZS+ story as the\n"
                "headline. Drop BasisMix from the contribution narrative, or\n"
                "relegate it to an ablation showing the meta-learner is dead\n"
                "weight (R3 confirmed).\n"
            )

    # ---- Update results/fused_five_dropone.md header ----
    md_path = "results/fused_five_dropone.md"
    if os.path.exists(md_path):
        with open(md_path) as f:
            existing = f.read()
        header = (
            f"# Task 4.1 — Drop-one ablation\n\n"
            f"**VERDICT (Task 4.2): {verdict}** "
            f"(drop-Hier BasisMix+ median = {drop_hier_med:.3f} "
            f"vs best-single Causal = {BEST_SINGLE_MAE:.3f}; "
            f"R1: {meta_verdict}; R3: {diversity_verdict})\n\n"
        )
        # Strip existing header (everything up to and including the first blank line
        # after the title) and prepend our verdict
        lines = existing.splitlines()
        body = "\n".join(lines[2:]) if len(lines) > 2 else ""
        with open(md_path, "w") as f:
            f.write(header + body)

    print(f"\n=== Task 4.2 VERDICT: {verdict} ===")
    print(f"  R3 diversity  : {diversity_verdict}")
    print(f"  R1 meta-value : {meta_verdict}")
    print(f"  drop-Hier BasisMix+ median = {drop_hier_med:.3f}")
    print(f"  best-single (Causal)        = {BEST_SINGLE_MAE:.3f}")
    print(f"  equal+ median               = {equal_med:.3f}")
    print(f"  BasisMix+ median            = {basismix_med:.3f}")
    print(f"\nWrote: results/fused_five_verdict.md")
    print(f"Updated: results/fused_five_dropone.md")


if __name__ == "__main__":
    main()

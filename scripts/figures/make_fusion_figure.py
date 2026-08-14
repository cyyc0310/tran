"""Direction-contribution figure for Section 6.10 (fusion motivation).

Reads results/directions_eval_summary.json and draws the five directions'
median MAE (zero-shot and ZS+), showing the complementary-prior structure that
the basis-mixture fusion combines. Output: figures/fusion_weights.png
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

HERE = Path(__file__).resolve().parent.parent.parent
d = json.loads((HERE / "results" / "directions_eval_summary.json").read_text())
dirs = ["rag", "phys_irm", "causal", "icl", "hier"]
labels = ["RAG\n(retrieval)", "Phys-IRM\n(physics)", "Causal\n(invariance)",
          "ICL\n(context)", "Hier\n(hierarchy)"]
zs = [d["directions"][k]["median_mae"] for k in dirs]
zsp = [d["directions"][k]["median_plus_mae"] for k in dirs]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(dirs))
w = 0.38
b1 = ax.bar(x - w/2, zs, w, color="#E69F00", label="zero-shot", alpha=0.9)
b2 = ax.bar(x + w/2, zsp, w, color="#0072B2", label="+ ZS+ calibration", alpha=0.9)
ax.axhline(d["baselines"]["median_persist_mae"], color="#999999", ls="--", lw=1,
           label=f"persistence floor ({d['baselines']['median_persist_mae']:.1f})")
for bars in (b1, b2):
    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Median MAE (gCO$_2$/kWh)")
ax.set_title("Five direction priors: complementary strengths (ZS+ equalizes them)")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.set_ylim(top=max(zs)*1.15)
fig.tight_layout()
out = HERE / "figures" / "fusion_weights.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print(f"[WRITE] {out}")

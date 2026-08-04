"""All-region MAE overview figure: every method's MAE on all 29 regions.

Bars: persistence / TransCIF-ZS / TransCIF-ZS+ (cross-domain, from
results/unified_eval_full.json, mean over 5 seeds). Markers: supervised
PatchTST upper bound (diamond) and CarbonCast-ZS where available
(9 representative regions, results/carboncast_analysis.json).
Stars mark the 27/29 regions where ZS+ is the lowest-MAE cross-domain method.
Output: figures/mae_overview_29regions.png/.pdf
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# Okabe-Ito, semantics fixed project-wide: blue = main model (ZS+),
# orange = our raw ZS, grey = persistence floor, green = CarbonCast
C_PERSIST = "#999999"
C_ZS = "#E69F00"
C_ZSP = "#0072B2"
C_CC = "#009E73"
C_SUP = "#000000"

with open("results/unified_eval_full.json") as f:
    ue = json.load(f)
rows = ue["results"] if isinstance(ue, dict) and "results" in ue else ue

grouped = defaultdict(list)
for r in rows:
    grouped[r["target"]].append(r)

regions = []
for tgt, rl in grouped.items():
    regions.append({
        "target": tgt,
        "mean_rs": float(np.mean([r["mean_rs"] for r in rl])),
        "persist": float(np.mean([r["persistence"]["mae"] for r in rl])),
        "sup": float(np.mean([r["patchtst_sup"]["mae"] for r in rl])),
        "zs": float(np.mean([r["transcif_zs"]["mae"] for r in rl])),
        "zsp": float(np.mean([r["transcif_zs_plus"]["mae"] for r in rl])),
        "zsp_std": float(np.std([r["transcif_zs_plus"]["mae"] for r in rl])),
    })
regions.sort(key=lambda d: d["mean_rs"])

with open("results/carboncast_analysis.json") as f:
    cc = json.load(f)
cc_zs = {r["region"]: r["cc_zeroshot"]["mae"] for r in cc}


def short(name):
    parts = name.split("_")
    return "_".join(parts[:2]) if name.startswith("UK") else name


n = len(regions)
x = np.arange(n)
w = 0.27

fig, ax = plt.subplots(figsize=(15.5, 5.6))

persist = [d["persist"] for d in regions]
zs = [d["zs"] for d in regions]
zsp = [d["zsp"] for d in regions]
zsp_err = [d["zsp_std"] for d in regions]
sup = [d["sup"] for d in regions]

ax.bar(x - w, persist, w, color=C_PERSIST, edgecolor="#6e6e6e", lw=0.6,
       label="Persistence (lag-24h)")
ax.bar(x, zs, w, color=C_ZS, edgecolor="#b57900", lw=0.6,
       label="TransCIF-ZS (config-only)")
ax.bar(x + w, zsp, w, yerr=zsp_err, capsize=1.5,
       error_kw={"lw": 0.7, "ecolor": "#333333"},
       color=C_ZSP, edgecolor="#004d78", lw=0.6,
       label="TransCIF-ZS+ (ours, test-time calib)")

ax.scatter(x, sup, marker="D", s=26, facecolor="white", edgecolor=C_SUP,
           lw=1.1, zorder=5, label="PatchTST supervised (upper bound)")

cc_x, cc_y = [], []
for i, d in enumerate(regions):
    key = next((k for k in cc_zs if d["target"].startswith(k.split("_South")[0])
                or k.startswith(d["target"])), None)
    if d["target"] in cc_zs:
        key = d["target"]
    if key is not None:
        cc_x.append(i)
        cc_y.append(cc_zs[key])
ax.scatter(cc_x, cc_y, marker="s", s=30, facecolor="white", edgecolor=C_CC,
           lw=1.3, zorder=5, label="CarbonCast-ZS (9 regions)")

# star = ZS+ is the lowest cross-domain MAE (vs persistence, ZS, CC-ZS)
for i, d in enumerate(regions):
    contenders = [d["persist"], d["zs"]]
    if i in cc_x:
        contenders.append(cc_y[cc_x.index(i)])
    top = max([d["persist"], d["zs"], d["zsp"], d["sup"]]
              + ([cc_y[cc_x.index(i)]] if i in cc_x else []))
    if d["zsp"] < min(contenders):
        ax.text(i, top + 2.5, "★", ha="center", va="bottom",
                fontsize=8, color=C_ZSP)
    else:
        ax.text(i, top + 2.5, "✕", ha="center", va="bottom",
                fontsize=7.5, color="#D55E00")

ax.set_xticks(x)
ax.set_xticklabels([short(d["target"]) for d in regions],
                   rotation=60, ha="right", fontsize=8.5)
ax.set_ylabel("CIF MAE (gCO$_2$/kWh)")
ax.set_title("All-method MAE across the 29-region LORO benchmark "
             "(regions sorted by mean renewable share; 5 seeds)\n"
             "★ = TransCIF-ZS+ has the lowest cross-domain MAE (27/29);  "
             "✕ = honest-selection exceptions (US_MISO, US_ERCO)",
             fontsize=10.5)
ax.grid(axis="y", alpha=0.22)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=5, loc="upper left",
          bbox_to_anchor=(0.0, 1.0), fontsize=8.5)
ax.set_ylim(0, max(max(persist), max(zs)) * 1.22)

mean_txt = (f"mean MAE  —  persistence {np.mean(persist):.1f}   "
            f"ZS {np.mean(zs):.1f}   ZS+ {np.mean(zsp):.1f} "
            f"(−{(1 - np.mean(zsp) / np.mean(zs)) * 100:.1f}% vs ZS)   "
            f"supervised {np.mean(sup):.1f}")
ax.text(0.99, 0.88, mean_txt, transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc"))

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"figures/mae_overview_29regions.{ext}", dpi=300,
                bbox_inches="tight")
print("saved figures/mae_overview_29regions.png/.pdf")
print(mean_txt)

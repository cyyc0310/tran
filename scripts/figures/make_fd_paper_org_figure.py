#!/usr/bin/env python
"""Paper organization figure for 2026-09-17-transcif-fd-paper-zh.md.

Three-phase reading map (SS1-3 -> SS4-6 -> SS7-9) plus the information-ladder
rail that runs through the whole paper. Pure diagram; reads no data files.

Usage:
    .venv-nemed312/bin/python scripts/figures/make_fd_paper_org_figure.py
Output:
    figures/fd_paper_org.png (300 dpi)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def pick_cjk_font():
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti",
              "Arial Unicode MS", "Songti SC", "Noto Sans CJK SC"]:
        if c in avail:
            return c
    return "sans-serif"


CJK = pick_cjk_font()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [CJK, "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})
print("CJK font:", CJK)

INK = "#1a1a1a"
MUT = "#5a615c"
PH = [
    ("问题与形式化", "§1–§3", "#eef4fa", "#2471a3"),
    ("方法 · 理论 · 实证", "§4–§6", "#edf6f0", "#1e8449"),
    ("落地 · 讨论 · 结论", "§7–§9", "#fdf4ec", "#e67e22"),
]
SECTIONS = [
    [("§1 引言", "零遥测日前 CIF 预测问题；\n信息层级主张与五项贡献"),
     ("§2 相关工作", "CarbonCast · CIF 基础模型 ·\n域泛化与 TSFM 的位置"),
     ("§3 问题形式化", "四层信息集 $I_\\mathrm{cfg}\\subset I_0\\subset I_+\\subset I_\\mathrm{lag}$\n与逐层预测契约")],
    [("§4 方法：FuelDecompNet", "十头物理分解 × 逐层条件化；\nZS+ 校准与滞后层免训练附加"),
     ("§5 理论", "定理 1：精确误差传播恒等式；\n定理 2：U 形消失＝架构依赖"),
     ("§6 实验", "层级定价 38.4 · LOJO 迁移 ·\nDunkelflaute 4.19× · NWP +6.4")],
    [("§7 中国零遥测部署层", "翻译层四组件（日历/进口 EF/\n配置表/信任门控）；山西/上海 15–18"),
     ("§8 讨论", "信息层级的正确读法；\n诚实边界与局限"),
     ("§9 结论", "政策披露到哪一层，\n方法升级到哪一层")],
]
TIERS = [
    ("$I_\\mathrm{cfg}$", "config 标量 + 公开气象/日历"),
    ("$I_0$", "+ 实时燃料份额流"),
    ("$I_+$", "+ CIF 历史（免训练校准）"),
    ("$I_\\mathrm{lag}$", "+ 滞后观测（零训练混合）"),
]

W, H = 11.6, 5.2
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

M = 0.16
GAP = 0.38
PTOP, PBOT = H - 0.14, 1.38
cw = (W - 2 * M - 2 * GAP) / 3


def panel_rect(i):
    x0 = M + i * (cw + GAP)
    return x0, x0 + cw


for i, (pt, pr, fill, edge) in enumerate(PH):
    x0, x1 = panel_rect(i)
    ax.add_patch(FancyBboxPatch((x0, PBOT), cw, PTOP - PBOT,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fill, ec=edge, lw=1.2, alpha=.55, zorder=1))
    ax.text((x0 + x1) / 2 - 0.30, PTOP - 0.21, pt, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=edge, zorder=3)
    ax.text(x1 - 0.12, PTOP - 0.21, pr, ha="right", va="center",
            fontsize=8.2, color=MUT, zorder=3)
    n = len(SECTIONS[i])
    ch, cg = 0.95, 0.13
    top = PTOP - 0.52
    for j, (t, d) in enumerate(SECTIONS[i]):
        ytop = top - j * (ch + cg)
        ybot = ytop - ch
        ax.add_patch(FancyBboxPatch((x0 + 0.14, ybot), cw - 0.28, ch,
                                    boxstyle="round,pad=0.02,rounding_size=0.07",
                                    fc="white", ec=edge, lw=1.0, zorder=3))
        ax.plot([x0 + 0.14, x0 + 0.14], [ybot + 0.12, ytop - 0.12], color=edge,
                lw=2.6, solid_capstyle="round", zorder=4)
        ax.text(x0 + 0.30, ytop - 0.21, t, fontsize=9.3, fontweight="bold",
                color=INK, va="center", zorder=4)
        ax.text(x0 + 0.30, ybot + 0.305, d, fontsize=7.6, color=MUT,
                va="center", ha="left", zorder=4, linespacing=1.45)
        if j < n - 1:
            ax.add_patch(FancyArrowPatch((x0 + cw / 2, ybot - 0.015),
                                         (x0 + cw / 2, ybot - cg + 0.015),
                                         arrowstyle="-|>", mutation_scale=9,
                                         color="#9aa59d", lw=1.0, zorder=3))
    if i < 2:
        xa, xb = x0 + cw, x0 + cw + GAP
        ax.add_patch(FancyArrowPatch((xa + 0.05, (PTOP + PBOT) / 2),
                                     (xb - 0.05, (PTOP + PBOT) / 2),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color="#8a948c", lw=1.6, zorder=3))

RB, RT = 0.16, 1.12
ax.add_patch(FancyBboxPatch((M, RB), W - 2 * M, RT - RB,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            fc="#e8f4f0", ec="#138d75", lw=1.2, alpha=.75, zorder=1))
ax.text(M + 0.18, RT - 0.155,
        "信息层级主线 — §3 形式定义 · §4 逐层构建 · §6 逐层定价 · §7 部署落地",
        fontsize=8.6, color="#0e5c4b", fontweight="bold", va="center", zorder=3)
tw, tg = 2.42, 0.50
tot = 4 * tw + 3 * tg
cx = M + ((W - 2 * M) - tot) / 2
cy = (RB + RT - 0.30) / 2 + 0.02
for k, (name, desc) in enumerate(TIERS):
    x0 = cx + k * (tw + tg)
    ax.add_patch(FancyBboxPatch((x0, cy - 0.21), tw, 0.44,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                fc="white", ec="#138d75", lw=1.0, zorder=3))
    ax.text(x0 + 0.14, cy + 0.085, name, fontsize=9.0, fontweight="bold",
            color="#0e5c4b", va="center", zorder=4)
    ax.text(x0 + 0.14, cy - 0.115, desc, fontsize=7.0, color=MUT,
            va="center", zorder=4)
    if k < 3:
        ax.add_patch(FancyArrowPatch((x0 + tw + 0.06, cy),
                                     (x0 + tw + tg - 0.06, cy),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color="#138d75", lw=1.4, zorder=3))
for i in range(3):
    x0, x1 = panel_rect(i)
    ax.plot([(x0 + x1) / 2, (x0 + x1) / 2], [RT + 0.02, PBOT - 0.02],
            ls=(0, (3, 3)), color="#9fb8ae", lw=1.1, zorder=0)

fig.savefig(FIG / "fd_paper_org.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved figures/fd_paper_org.png")

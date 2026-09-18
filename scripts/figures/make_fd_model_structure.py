#!/usr/bin/env python
"""FuelDecompNet model-structure figure (matches src/transcif/models/fuel_decomp.py
and paper section 4 as of the 2026-09-17 FD revision).

Left-to-right dataflow: three deployment-available input groups -> small
encoders -> ten-head physics decomposition + aggregate head -> closed-form
physics synthesis -> zero-parameter deterministic router / level anchor -> output.

Usage:
    .venv-nemed/bin/python scripts/figures/make_fd_model_structure.py
Output:
    figures/fd_model_structure.png (300 dpi)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})
print("CJK font:", CJK)

INK = "#20261f"
BODY = "#3a423c"
MUT = "#6b736c"
BLUE, GREEN, TEAL = "#2471a3", "#1e8449", "#138d75"
PURPLE, ORANGE, GREY = "#7d3c98", "#e67e22", "#8a8a8a"

# face / edge pairs
IN_FC, IN_EC = "#edf1f7", "#7f9fd4"      # inputs
ENC_FC, ENC_EC = "#ece7f7", "#7d3c98"    # encoders
HEAD_FC, HEAD_EC = "#e3ecf9", "#2471a3"  # fuel heads
AGG_FC, AGG_EC = "#e6f4f1", "#138d75"    # aggregate head
PHY_FC, PHY_EC = "#e9f7ee", "#1e8449"    # physics
RTE_FC, RTE_EC = "#fdf3e0", "#e67e22"    # router
ANC_FC, ANC_EC = "#f3edf9", "#7d3c98"    # anchor
OUT_FC, OUT_EC = "#dcefe6", "#0f5132"    # output
NOTE_FC, NOTE_EC = "#f7f5fb", "#b9aed6"


def box(ax, x, y, w, h, title, lines, fc, ec, tag=None,
        fs=6.8, tfs=8.6, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.9",
        fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + 1.6, y + h - 2.8, title, fontsize=tfs, fontweight="bold",
            color=INK, va="center", ha="left", zorder=4)
    if tag:
        ax.text(x + w - 1.6, y + h - 2.8, tag, fontsize=6.3,
                style="italic", color=ec, va="center", ha="right", zorder=4)
    yy = y + h - 6.4
    for ln in lines:
        ax.text(x + 1.6, yy, ln, fontsize=fs, color=BODY, va="center",
                ha="left", zorder=4)
        yy -= 3.0


def ar(ax, x0, y0, x1, y1, c, lw=1.5, ls="-", ms=11):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                linestyle=ls, mutation_scale=ms,
                                shrinkA=0, shrinkB=0), zorder=2)


def seg(ax, pts, c, lw=1.4, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, solid_capstyle="round", zorder=2)


W, H = 160, 94
fig, ax = plt.subplots(figsize=(16, 9.6))
ax.set_xlim(0, W)
ax.set_ylim(-5, H)
ax.axis("off")

# ------------------------------------------------------------------ title
ax.text(W / 2, 92.3, "FuelDecompNet 模型结构", ha="center", va="center",
        fontsize=14, fontweight="bold", color=INK)
ax.text(W / 2, 89.0,
        "十头物理分解 × 逐层条件化 × 零参数确定性路由 · L=336 h 输入 → H=24 h 日前 CIF · <25k 参数",
        ha="center", va="center", fontsize=9.5, color=MUT)

# ------------------------------------------------------- column headers
for cx, t, c in [(13.5, "输入（部署可得）", GREY), (40.5, "小容量编码", PURPLE),
                 (77, "十头物理分解 + 聚合头", BLUE), (115, "物理合成（闭式）", GREEN),
                 (145.5, "路由 · 锚定 · 输出", ORANGE)]:
    ax.text(cx, 86.0, t, ha="center", va="center", fontsize=7.8,
            fontweight="bold", color=c)

# ------------------------------------------------------------------ col A
box(ax, 2, 64, 23, 20, "config c（16 维）", [
    "mean_rs · ef_nr · 10 燃料份额",
    "年均风 CF · 年均晴空指数",
    "has_fuel · |lat|/60",
    "—— I_cfg 层即可得（公开）",
], IN_FC, IN_EC)

box(ax, 2, 34, 23, 26, "历史流（仅 I_0 层）", [
    "x_rs  (L=336h)  可再生份额",
    "x_fuel (L×10)  逐燃料份额",
    "x_weather (L×10)  天气",
    "hist_mask 模式开关",
    "冷模式 dropout p=0.3",
    "→ 整窗置零（含天气）",
], IN_FC, IN_EC)

box(ax, 2, 10, 23, 22, "未来通道（全层级可得）", [
    "fut_weather (H×10) 预报天气",
    "fut_exog (H×E)",
    "晴空 / 风 CF / 日历相位",
    "部署时来自公开 NWP",
], IN_FC, IN_EC)

ax.add_patch(FancyBboxPatch((2, 0), 23, 7.2,
                            boxstyle="round,pad=0.02,rounding_size=0.9",
                            fc=IN_FC, ec=IN_EC, lw=1.1, zorder=3))
ax.text(13.5, 3.6, "EF 向量 ef_f（10 燃料，查表）", fontsize=7.0,
        color=INK, va="center", ha="center", fontweight="bold", zorder=4)

# ------------------------------------------------------------------ col B
box(ax, 30, 66, 21, 16, "config 编码 MLP", [
    "16 → 32 → 32（刻意小容量）",
    "深度条件化三次被证明有害",
], ENC_FC, ENC_EC)

box(ax, 30, 10, 21, 22, "未来上下文编码器", [
    "Conv1d(3)+GELU",
    "→ Conv1d(5)+GELU → 16 维",
    "零初始化投影头：初始=基线",
    "保留时序：爬坡 / 鸭子曲线",
], ENC_FC, ENC_EC)

box(ax, 30, 33, 21, 12, "一套权重 · 两种模式", [
    "冷模式 dropout p=0.3：每窗独立",
    "丢弃历史 → I_cfg 与 I_0 共享参数",
], NOTE_FC, NOTE_EC, fs=6.4, tfs=7.4)

# ------------------------------------------------------------------ col C
heads_y = {"solar": 69, "wind": 52.3, "base": 35.6, "therm": 18.9, "agg": 2.2}
HH = 15.0
box(ax, 59, heads_y["solar"], 36, HH, "Solar 头", [
    "$\\hat{s}$ = 水平(cfg/历史) × [astro(h)/日均astro]",
    "　　× (1 + 0.4·tanh(MLP(未来天气)))",
], HEAD_FC, HEAD_EC, tag="天文通道主导")

box(ax, 59, heads_y["wind"], 36, HH, "Wind 头", [
    "$\\hat{s}$ = 水平 × [wcf(h)/wcf参考] × 调制",
    "norm∈[0.2,3]；参考=0.7·近周+0.3·年均",
    "干旱锚定：regime 远小于年均 → 回归年均",
], HEAD_FC, HEAD_EC, tag="防爆炸")

box(ax, 59, heads_y["base"], 36, HH, "Baseload 头 ×5", [
    "（核电/水电/生物质/进口/其他）",
    "水平(历史168h 或 config) + 掩码×(先验+修正)",
], HEAD_FC, HEAD_EC, tag="零份额不幻觉")

box(ax, 59, heads_y["therm"], 36, HH, "Thermal 头 ×3", [
    "$T$ = 1 - Σ 非可调度（调度竞争残差）",
    "split = softmax(log(cfg+0.02)+掩码×修正)",
], HEAD_FC, HEAD_EC, tag="零初始化=配置先验")

box(ax, 59, heads_y["agg"], 36, HH, "聚合头（两用）", [
    "DLinear(rs) + logit(mean_rs) 锚定 + 持久门(历史)",
    "路径A：高风/高水电聚合 CIF",
    "路径B：份额一致性辅助信号",
], AGG_FC, AGG_EC, tag="高风/高水电更稳")

# ------------------------------------------------------------------ col D
box(ax, 101, 60, 28, 14, "份额组装（10 燃料）", [
    "clamp [0,0.95] → 沿燃料轴重归一",
    "solar1 + wind1 + base5 + thermal3",
], PHY_FC, PHY_EC, fs=6.7)

box(ax, 101, 36, 28, 20, "物理合成层（闭式）", [
    "$\\mathrm{CIF}_{fuel}$ = $\\sum_f \\hat{s}_f\\,ef_f$ × (1±0.35·tanh(EF校正))",
    "进口 EF 专用通路 ±0.9（邻网日历相位）",
    "有界动态碳流残差 ≤220·tanh（零初始化）",
], PHY_FC, PHY_EC, fs=6.7)

box(ax, 106, 6, 23, 17, "聚合路径合成（2 燃料）", [
    "$\\mathrm{CIF}_{agg}$ = rs·ef_r + (1-rs)·ef_nr",
    "定理 1 二分类特例，",
    "增益 $L_T$=|ef_r-ef_nr| 已知",
], AGG_FC, AGG_EC, fs=6.7)

# ------------------------------------------------------------------ col E
box(ax, 133, 46, 25, 24, "确定性结构路由（零参数）", [
    "route = has_fuel · σ(20(τ-wind_cfg))",
    "　　　　 · σ(30(0.5-hydro))，τ=1.1",
    "零遥测 / 高风 / 高水电 → 聚合路径",
    "只读 config，部署前即可判定路径",
], RTE_FC, RTE_EC, fs=6.7)

box(ax, 133, 22, 25, 20, "水平锚定（形状/水平解耦）", [
    "历史模式：cif += g·((1-近48h均值rs)",
    "　　·ef_nr - 模型水平)，门控近锚定",
    "冷模式：锚 config mean_rs",
    "（月度公报滞后值，零遥测合法）",
], ANC_FC, ANC_EC, fs=6.7)

box(ax, 133, 2, 25, 16, "输出", [
    "CIF (B, 24h) 日前曲线",
    "逐燃料份额 (B, 24, 10)",
    "定理 1：误差逐燃料可归因",
], OUT_FC, OUT_EC, fs=6.9, lw=1.8)

# ------------------------------------------------------------------ wiring
CY = {"solar": heads_y["solar"] + HH / 2, "wind": heads_y["wind"] + HH / 2,
      "base": heads_y["base"] + HH / 2, "therm": heads_y["therm"] + HH / 2,
      "agg": heads_y["agg"] + HH / 2}

# input arrows
ar(ax, 25, 74, 30, 74, BLUE)                       # config -> cfg mlp
ar(ax, 25, 21, 30, 21, BLUE)                       # future -> ctx encoder
seg(ax, [(25, 47), (54.0, 47)], GREY, lw=1.5)      # history -> grey rail

# rails in the 51..59 gap: blue(future ctx) / grey(history) / purple(config)
seg(ax, [(51.8, 9.7), (51.8, CY["solar"])], BLUE, lw=1.3)
seg(ax, [(54.0, 8.7), (54.0, CY["solar"] - 1.0)], GREY, lw=1.3)
seg(ax, [(56.4, 10.7), (56.4, CY["solar"] + 1.0)], PURPLE, lw=1.2, ls="--")
seg(ax, [(51, 21), (51.8, 21)], BLUE, lw=1.3)
seg(ax, [(51, 74), (56.4, 74)], PURPLE, lw=1.2, ls="--")
for k, yy in CY.items():
    ar(ax, 51.8, yy, 59, yy, BLUE, lw=1.1, ms=8)
    ar(ax, 54.0, yy - 1.0, 59, yy - 1.0, GREY, lw=1.1, ms=8)
    ar(ax, 56.4, yy + 1.0, 59, yy + 1.0, PURPLE, lw=1.0, ms=8, ls="--")
for x, lab, c in [(51.8, "未来上下文", BLUE), (54.0, "历史水平(I_0)", GREY),
                  (56.4, "config 条件化", PURPLE)]:
    ax.text(x, 78.2, lab, rotation=90, fontsize=6.0, color=c,
            ha="center", va="bottom", zorder=4)

# heads -> share assembly
for key, ye in [("solar", 70), ("wind", 68), ("base", 66), ("therm", 64)]:
    ar(ax, 95, CY[key], 101, ye, GREEN, lw=1.4)
ar(ax, 115, 60, 115, 56.2, GREEN, lw=1.6)          # assembly -> physics
ar(ax, 95, CY["agg"], 106, 14.5, TEAL, lw=1.4)     # agg head -> cif_agg

# EF vector -> physics layer (under the bottom row, up at x=105)
seg(ax, [(25, 1.2), (105, 1.2)], GREEN, lw=1.5)
ar(ax, 105, 1.2, 105, 35.8, GREEN, lw=1.5)

# synthesis -> router
ar(ax, 129, 48, 133, 58, GREEN, lw=1.6)
ar(ax, 129, 14.5, 133, 52, TEAL, lw=1.6)
# router -> anchor -> output
ar(ax, 145.5, 46, 145.5, 42.2, ORANGE, lw=1.7)
ar(ax, 145.5, 22, 145.5, 18.2, GREEN, lw=1.7)

# ------------------------------------------------------------------ footer
ax.text(2, -2.6,
        "设计纪律：全模型 <25k 参数；深度条件化（FiLM 30×、超网络、融合元学习器）三次被证明有害；"
        "所有动态修正零初始化——训练从物理先验出发；冷模式 dropout 使同一套权重服务 I_cfg 与 I_0，零遥测部署无需改动模型。",
        fontsize=7.0, color=MUT, ha="left", va="center")

fig.savefig(FIG / "fd_model_structure.png", bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print("saved figures/fd_model_structure.png")

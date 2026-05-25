"""
Fig 6. 2022 하락장 vs 2024~2026 상승장 비교
'상승장에선 방어적, 하락장에선 강함'
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})

# ── 데이터 ────────────────────────────────────────────────────────
with open("outputs/results/backtest_2022_results.json") as f:
    bear = json.load(f)

with open("outputs/results/backtest_results.json") as f:
    bull = json.load(f)

strategies = ["Buy & Hold", "60/40", "MA Crossover", "Conv1D+LSTM (ours)"]
labels     = ["Buy &\nHold", "60/40", "MA\nCrossover", "Ours\n◀"]

cum_bear  = [bear[s]["cum"]     * 100 for s in strategies]
mdd_bear  = [bear[s]["mdd"]     * 100 for s in strategies]
cum_bull  = [bull[s]["cum_ret"] * 100 for s in strategies]
mdd_bull  = [bull[s]["mdd"]     * 100 for s in strategies]

BLUE  = "#2980B9"
GRAY  = "#95A5A6"
RED   = "#E74C3C"
GREEN = "#27AE60"

def bar_colors(strategies, highlight="Conv1D+LSTM (ours)"):
    return [BLUE if s == highlight else GRAY for s in strategies]

fig = plt.figure(figsize=(14, 9))
fig.suptitle(
    "시장 국면별 전략 비교: 상승장에선 방어적, 하락장에선 강함",
    fontsize=15, fontweight="bold", y=0.98
)

x = np.arange(len(labels))
w = 0.55

# ── 서브타이틀 텍스트 박스 ─────────────────────────────────────────
ax_title1 = fig.add_axes([0.05, 0.88, 0.42, 0.05])
ax_title1.axis("off")
ax_title1.text(0.5, 0.5, "2022 하락장  (SPY -18.6%)", ha="center", va="center",
               fontsize=12, fontweight="bold", color=RED,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor=RED, linewidth=1.5))

ax_title2 = fig.add_axes([0.53, 0.88, 0.42, 0.05])
ax_title2.axis("off")
ax_title2.text(0.5, 0.5, "2024~2026 상승장  (SPY +49.9%)", ha="center", va="center",
               fontsize=12, fontweight="bold", color=GREEN,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAF7EA", edgecolor=GREEN, linewidth=1.5))

# ── 누적수익률 ────────────────────────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
colors1 = bar_colors(strategies)
bars1 = ax1.bar(x, cum_bear, color=colors1, alpha=0.85, width=w)
for bar, val in zip(bars1, cum_bear):
    ypos = bar.get_height() - 1.5 if val < 0 else bar.get_height() + 0.3
    va   = "top" if val < 0 else "bottom"
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{val:.1f}%", ha="center", va=va, fontsize=10, fontweight="bold")
bars1[-1].set_edgecolor("gold"); bars1[-1].set_linewidth(2.5)
ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("누적 수익률 (%)"); ax1.set_title("누적 수익률", fontsize=11)
ax1.set_ylim(min(cum_bear) * 1.4, max(cum_bear) * 1.5 + 2)

ax2 = fig.add_subplot(2, 2, 2)
colors2 = bar_colors(strategies)
bars2 = ax2.bar(x, cum_bull, color=colors2, alpha=0.85, width=w)
for bar, val in zip(bars2, cum_bull):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
bars2[-1].set_edgecolor("gold"); bars2[-1].set_linewidth(2.5)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("누적 수익률 (%)"); ax2.set_title("누적 수익률", fontsize=11)

# ── MDD ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
colors3 = bar_colors(strategies)
bars3 = ax3.bar(x, [abs(v) for v in mdd_bear], color=colors3, alpha=0.85, width=w)
for bar, val in zip(bars3, mdd_bear):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{abs(val):.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
bars3[-1].set_edgecolor("gold"); bars3[-1].set_linewidth(2.5)
ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel("Max Drawdown (%)"); ax3.set_title("MDD  (낮을수록 좋음)", fontsize=11)

ax4 = fig.add_subplot(2, 2, 4)
colors4 = bar_colors(strategies)
bars4 = ax4.bar(x, [abs(v) for v in mdd_bull], color=colors4, alpha=0.85, width=w)
for bar, val in zip(bars4, mdd_bull):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{abs(val):.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
bars4[-1].set_edgecolor("gold"); bars4[-1].set_linewidth(2.5)
ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=9)
ax4.set_ylabel("Max Drawdown (%)"); ax4.set_title("MDD  (낮을수록 좋음)", fontsize=11)

# ── 범례 및 요약 텍스트 ───────────────────────────────────────────
ours_patch = mpatches.Patch(color=BLUE, label="Conv1D+LSTM (ours)")
gray_patch  = mpatches.Patch(color=GRAY, label="기타 전략")
fig.legend(handles=[ours_patch, gray_patch], loc="lower center",
           ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01),
           framealpha=0.9)

# 핵심 인사이트 텍스트
insight = (
    "하락장(2022):  MDD  -4.9%  vs  Buy&Hold  -20.5%  →  낙폭 76% 감소\n"
    "상승장(2024~): MDD  -5.2%  vs  Buy&Hold  -17.0%  →  꾸준한 하락 방어"
)
fig.text(0.5, 0.055, insight, ha="center", va="bottom", fontsize=10,
         color="#2C3E50", style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#BDC3C7"))

plt.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12, hspace=0.45, wspace=0.3)
plt.savefig("outputs/figures/fig6_market_comparison.png", bbox_inches="tight")
plt.close()
print("Fig6 저장: outputs/figures/fig6_market_comparison.png")

"""
Fig 6. 2022 하락장 vs 2024~2026 상승장 비교
'상승장에선 방어적, 하락장에선 강함'
전략: Buy & Hold / EW 1/N / 60/40 / MA Crossover / Regime Momentum Tilt
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── 데이터 로드 ───────────────────────────────────────────────────
with open("outputs/results/backtest_2022_results.json") as f:
    bear = json.load(f)

with open("outputs/results/backtest_results.json") as f:
    bt = json.load(f)

with open("outputs/results/backtest_regime_momentum_results.json") as f:
    bt_rmt = json.load(f)

# bull 기간: Regime Momentum Tilt는 backtest_regime_momentum_results.json 사용
bull = {**bt, "Regime Momentum Tilt": bt_rmt["Regime Momentum Tilt"]}

# 2022 bear에서 "Conv1D+LSTM (SPY/Cash)" → bull에서는 "Conv1D+LSTM (ours)"로 동일 전략
bear_keys = ["Buy & Hold", "EW (1/N)", "60/40", "MA Crossover",
             "Conv1D+LSTM (SPY/Cash)", "Regime Momentum Tilt"]
bull_keys = ["Buy & Hold", "EW (1/N)", "60/40", "MA Crossover",
             "Conv1D+LSTM (ours)", "Regime Momentum Tilt"]
labels    = ["Buy &\nHold", "EW\n1/N", "60/40", "MA\nCrossover", "SPY/Cash\n(ours) ◀", "Regime\nTilt"]

cum_bear  = [bear[s]["cum"]     * 100 for s in bear_keys]
mdd_bear  = [bear[s]["mdd"]     * 100 for s in bear_keys]
cum_bull  = [bull[s]["cum_ret"] * 100 for s in bull_keys]
mdd_bull  = [bull[s]["mdd"]     * 100 for s in bull_keys]

BLUE   = "#2980B9"
ORANGE = "#E67E22"
GREEN  = "#27AE60"
GRAY   = "#95A5A6"
RED    = "#E74C3C"

def bar_colors(keys):
    return [BLUE if "SPY/Cash" in s or "(ours)" in s else
            ORANGE if "Regime" in s else
            GREEN if "EW" in s else GRAY for s in keys]

fig = plt.figure(figsize=(14, 9))
fig.suptitle(
    "시장 국면별 전략 비교: 상승장에선 방어적, 하락장에선 강함",
    fontsize=15, fontweight="bold", y=0.98
)

x = np.arange(len(labels))
w = 0.55

# ── 서브타이틀 텍스트 박스 ────────────────────────────────────────
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
bars1 = ax1.bar(x, cum_bear, color=bar_colors(bear_keys), alpha=0.85, width=w)
for bar, val in zip(bars1, cum_bear):
    ypos = bar.get_height() - 1.5 if val < 0 else bar.get_height() + 0.3
    va   = "top" if val < 0 else "bottom"
    ax1.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{val:.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("누적 수익률 (%)"); ax1.set_title("누적 수익률", fontsize=11)
ax1.set_ylim(min(cum_bear) * 1.5, max(max(cum_bear) * 1.5, 3))

ax2 = fig.add_subplot(2, 2, 2)
bars2 = ax2.bar(x, cum_bull, color=bar_colors(bull_keys), alpha=0.85, width=w)
for bar, val in zip(bars2, cum_bull):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("누적 수익률 (%)"); ax2.set_title("누적 수익률", fontsize=11)

# ── MDD ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
bars3 = ax3.bar(x, [abs(v) for v in mdd_bear], color=bar_colors(bear_keys), alpha=0.85, width=w)
best3 = [abs(v) for v in mdd_bear].index(min([abs(v) for v in mdd_bear]))
bars3[best3].set_edgecolor("gold"); bars3[best3].set_linewidth(2.5)
for bar, val in zip(bars3, mdd_bear):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{abs(val):.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel("Max Drawdown (%)"); ax3.set_title("MDD  (낮을수록 좋음)", fontsize=11)

ax4 = fig.add_subplot(2, 2, 4)
bars4 = ax4.bar(x, [abs(v) for v in mdd_bull], color=bar_colors(bull_keys), alpha=0.85, width=w)
best4 = [abs(v) for v in mdd_bull].index(min([abs(v) for v in mdd_bull]))
bars4[best4].set_edgecolor("gold"); bars4[best4].set_linewidth(2.5)
for bar, val in zip(bars4, mdd_bull):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{abs(val):.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=9)
ax4.set_ylabel("Max Drawdown (%)"); ax4.set_title("MDD  (낮을수록 좋음)", fontsize=11)

# ── 범례 및 요약 텍스트 ──────────────────────────────────────────
lstm_patch = mpatches.Patch(color=BLUE,   label="Conv1D+LSTM SPY/Cash (ours)")
ew_patch   = mpatches.Patch(color=GREEN,  label="EW 1/N (논문 벤치마크)")
rmt_patch  = mpatches.Patch(color=ORANGE, label="Regime Momentum Tilt")
gray_patch = mpatches.Patch(color=GRAY,   label="기타 전략")
fig.legend(handles=[lstm_patch, ew_patch, rmt_patch, gray_patch], loc="lower center",
           ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

# 핵심 인사이트 — 실제 수치 자동 계산
lstm_mdd_bear = abs(bear["Conv1D+LSTM (SPY/Cash)"]["mdd"]) * 100
bnh_mdd_bear  = abs(bear["Buy & Hold"]["mdd"]) * 100
lstm_mdd_bull = abs(bull["Conv1D+LSTM (ours)"]["mdd"]) * 100
bnh_mdd_bull  = abs(bull["Buy & Hold"]["mdd"]) * 100
reduction     = (1 - lstm_mdd_bear / bnh_mdd_bear) * 100

insight = (
    f"하락장(2022): Conv1D+LSTM  MDD -{lstm_mdd_bear:.1f}%  vs  Buy&Hold -{bnh_mdd_bear:.1f}%  →  낙폭 {reduction:.0f}% 감소\n"
    f"상승장(2024~): Conv1D+LSTM  MDD -{lstm_mdd_bull:.1f}%  vs  Buy&Hold -{bnh_mdd_bull:.1f}%  →  꾸준한 위험 관리"
)
fig.text(0.5, 0.055, insight, ha="center", va="bottom", fontsize=10,
         color="#2C3E50", style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#BDC3C7"))

plt.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12, hspace=0.45, wspace=0.3)
plt.savefig("outputs/figures/fig6_market_comparison.png", bbox_inches="tight")
plt.close()
print("Fig6 저장: outputs/figures/fig6_market_comparison.png")

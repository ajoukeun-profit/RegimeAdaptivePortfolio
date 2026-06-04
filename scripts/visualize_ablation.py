"""
Fig 07. Ablation Study
포트폴리오 최적화 관점: 각 구성요소가 MDD/Calmar에 어떻게 기여하는가
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})

# ── 데이터 ────────────────────────────────────────────────────────
# Configuration | Key Change | Cum(%) | Sharpe | MDD(%) | Calmar
ABLATION = [
    ("Buy & Hold",
     "Baseline\n(no optimization)",
     49.9, 1.07, -17.0, 1.26, "#7F8C8D"),

    ("EW 1/N",
     "Static diversified\nbenchmark",
     50.9, 1.41, -8.8, 2.47, "#27AE60"),

    ("Regime-Agnostic MVO",
     "MVO without\nregime conditioning",
     64.8, 1.11, -20.8, 1.30, "#E67E22"),

    ("DL Regime\nSPY/Cash",
     "Classifier signal\n+ simple policy",
     21.9, 0.73, -7.4, 1.35, "#2980B9"),

    ("Regime-MVO\n(ours)",
     "Regime signal\n+ MVO policy",
     35.3, 1.10, -7.2, 2.16, "#E74C3C"),

    ("Oracle\n(HMM labels)",
     "Pseudo-label\nupper bound",
     41.6, 1.16, -6.2, 2.91, "#8E44AD"),
]

COLORS = [row[6] for row in ABLATION]

fig = plt.figure(figsize=(16, 8.6))
fig.suptitle(
    "Ablation and Benchmark Check: Static vs Dynamic Components\n"
    "(Test: 2024.04 ~ 2026.05  |  Primary interpretation: downside-risk control)",
    fontsize=12, fontweight="bold", y=0.98,
)

# ── 상단: 표 ─────────────────────────────────────────────────────
ax_tbl = fig.add_axes([0.03, 0.56, 0.94, 0.38])
ax_tbl.axis("off")

col_headers = ["#", "Configuration", "Key Change",
               "Cum. Return", "Sharpe", "MDD", "Calmar"]
col_widths   = [0.3, 1.6, 2.0, 0.9, 0.7, 0.8, 0.8]
total_w = sum(col_widths)
xs = [sum(col_widths[:i]) for i in range(len(col_headers))]
row_h = 0.13; header_h = 0.14

ax_tbl.set_xlim(0, total_w); ax_tbl.set_ylim(0, 1.0)

# 헤더
for j, (hdr, cw, x0) in enumerate(zip(col_headers, col_widths, xs)):
    ax_tbl.add_patch(plt.Rectangle((x0, 1.0 - header_h), cw, header_h,
                                   facecolor="#1B2631", edgecolor="white", lw=0.5))
    ax_tbl.text(x0 + cw/2, 1.0 - header_h/2, hdr,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")

# 데이터 행
for i, (cfg, change, cum, sharpe, mdd, calmar, color) in enumerate(ABLATION):
    y0     = 1.0 - header_h - (i + 1) * row_h
    is_ours   = "ours" in cfg
    is_oracle = "Oracle" in cfg
    is_ew = "EW" in cfg
    bg = "#EBF5FB" if is_ours else ("#F5EEF8" if is_oracle else ("#F8F9FA" if i % 2 == 0 else "white"))
    if is_ew:
        bg = "#EAF7EA"

    row_vals = [str(i+1), cfg, change,
                f"{cum:.1f}%", f"{sharpe:.2f}", f"{mdd:.1f}%", f"{calmar:.2f}"]

    # MDD/Calmar 색상
    mdd_color    = "#1E8449" if mdd > -10 else "#C0392B"
    calmar_color = "#1E8449" if calmar >= 2.0 else ("#B7770D" if calmar >= 1.5 else "#C0392B")

    val_colors = [None, None, None, None, None, mdd_color, calmar_color]

    for j, (val, cw, x0, vc) in enumerate(zip(row_vals, col_widths, xs, val_colors)):
        ax_tbl.add_patch(plt.Rectangle((x0, y0), cw, row_h,
                                       facecolor=bg, edgecolor="#D5D8DC", lw=0.4))
        fw = "bold" if (is_ours or is_oracle or is_ew) else "normal"
        fc = vc if vc else ("#1A5276" if is_ours else ("#6C3483" if is_oracle else ("#1E8449" if is_ew else "#2C3E50")))
        if j == 1:
            ax_tbl.add_patch(plt.Rectangle((x0 + 0.04, y0 + 0.04),
                                           0.07, row_h - 0.08,
                                           facecolor=color, edgecolor="none"))
            ax_tbl.text(x0 + 0.16, y0 + row_h/2, val,
                        ha="left", va="center", fontsize=8, fontweight=fw,
                        color=fc, linespacing=1.2)
        else:
            ax_tbl.text(x0 + cw/2, y0 + row_h/2, val,
                        ha="center", va="center", fontsize=8.5,
                        fontweight=fw, color=fc, linespacing=1.2)

ax_tbl.add_patch(plt.Rectangle((0, 1.0 - header_h - len(ABLATION) * row_h),
                                total_w, header_h + len(ABLATION) * row_h,
                                fill=False, edgecolor="#2C3E50", lw=1.2))


# ── 하단: 막대 3패널 ──────────────────────────────────────────────
names_short = ["B&H", "EW\n1/N", "Agnostic\nMVO", "DL Regime\nSPY/Cash", "Regime-\nMVO\n(ours)", "Oracle\n(HMM)"]
mdds    = [abs(r[4]) for r in ABLATION]
calmars = [r[5] for r in ABLATION]
sharpes = [r[3] for r in ABLATION]
x = np.arange(len(ABLATION))

for pi, (vals, title, higher) in enumerate([
    (sharpes, "Sharpe Ratio\n(higher = better)", True),
    (mdds,    "Max Drawdown (%)\n(lower = better)", False),
    (calmars, "Calmar Ratio\n(higher = better)", True),
]):
    ax = fig.add_axes([0.05 + pi * 0.32, 0.05, 0.27, 0.36])
    bars = ax.bar(x, vals, color=COLORS, alpha=0.85, width=0.6)

    best = min(vals) if not higher else max(vals)
    for bar, val, c in zip(bars, vals, COLORS):
        if abs(val - best) < 0.001:
            bar.set_edgecolor("gold"); bar.set_linewidth(2.5)
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals) * 0.02,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=c)

    ax.set_xticks(x)
    ax.set_xticklabels(names_short, fontsize=7.5)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

Path("outputs/figures/final").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/final/fig07_ablation.png", bbox_inches="tight")
plt.close()
print("저장: outputs/figures/final/fig07_ablation.png")

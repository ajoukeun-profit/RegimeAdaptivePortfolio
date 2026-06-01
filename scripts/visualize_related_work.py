"""
Fig 9. 관련 연구 비교표 — RegimeFolio Table I/II 스타일
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})

# ── 데이터 정의 ────────────────────────────────────────────────────
# (study, method, Regime Awareness, Learned DL Classifier, Multi-Asset, MVO/Opt, Interpretability)
# value: "Y" | "N" | "P" (partial/implicit)

ROWS = [
    ("Kim et al.\n(2019) [1]",
     "HMM Regime Labeling\n+ Equal-Weight Portfolio",
     "Y", "N", "N", "N", "Y"),

    ("Jiang et al.\n(2017) [2]",
     "Deep Reinforcement Learning\n(EIIE, end-to-end)",
     "N", "N", "Y", "N", "N"),

    ("Chen et al.\n(2021) [3]",
     "Deep RL + Attention\n(implicit regime)",
     "P", "N", "Y", "N", "N"),

    ("Zhang et al.\n(2025) [4]\nRegimeFolio",
     "VIX Rule-based Regime\n+ RF/GB Ensemble + MVO",
     "Y", "N", "Y", "Y", "Y"),

    ("Ours",
     "Conv1D+LSTM Regime Classifier\n+ Regime-Conditioned MVO",
     "Y", "Y", "Y", "Y", "Y"),
]

COL_HEADERS = [
    "Study", "Method",
    "Regime\nAwareness",
    "Learned DL\nClassifier",
    "Multi-\nAsset",
    "MVO /\nOptimization",
    "Inter-\npretability",
]

# ── 셀 스타일 ──────────────────────────────────────────────────────
CELL_TEXT = {"Y": "O", "N": "X", "P": "~"}
CELL_BG   = {
    "Y": "#D5F5E3",   # 연초록
    "N": "#FADBD8",   # 연빨강
    "P": "#FEF9E7",   # 연노랑
}
CELL_FG   = {"Y": "#1E8449", "N": "#C0392B", "P": "#B7770D"}

# ── 레이아웃 ───────────────────────────────────────────────────────
n_rows = len(ROWS)
n_cols = len(COL_HEADERS)
col_widths = [1.4, 2.8, 1.0, 1.0, 0.8, 1.0, 1.0]  # relative
total_w    = sum(col_widths)
fig_w      = 14
fig_h      = 4.8

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")
ax.set_xlim(0, total_w)
ax.set_ylim(0, n_rows + 1.2)

row_h    = 0.88
header_h = 1.1
xs = [sum(col_widths[:i]) for i in range(n_cols)]  # left edges

# ── 헤더 ─────────────────────────────────────────────────────────
for j, (header, cw, x0) in enumerate(zip(COL_HEADERS, col_widths, xs)):
    ax.add_patch(plt.Rectangle((x0, n_rows * row_h), cw, header_h,
                                facecolor="#1B2631", edgecolor="white", linewidth=0.5))
    ax.text(x0 + cw / 2, n_rows * row_h + header_h / 2, header,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", linespacing=1.3)

# ── 데이터 행 ────────────────────────────────────────────────────
for i, row in enumerate(ROWS):
    y0      = (n_rows - 1 - i) * row_h
    is_ours = row[0] == "Ours"
    row_bg  = "#EBF5FB" if is_ours else ("#F8F9FA" if i % 2 == 0 else "white")

    study, method = row[0], row[1]
    flags = row[2:]   # 5개 feature flags

    # Study 셀
    ax.add_patch(plt.Rectangle((xs[0], y0), col_widths[0], row_h,
                                facecolor=row_bg, edgecolor="#D5D8DC", linewidth=0.5))
    ax.text(xs[0] + col_widths[0] / 2, y0 + row_h / 2, study,
            ha="center", va="center", fontsize=8,
            fontweight="bold" if is_ours else "normal",
            color="#1A5276" if is_ours else "#2C3E50", linespacing=1.3)

    # Method 셀
    ax.add_patch(plt.Rectangle((xs[1], y0), col_widths[1], row_h,
                                facecolor=row_bg, edgecolor="#D5D8DC", linewidth=0.5))
    ax.text(xs[1] + col_widths[1] / 2, y0 + row_h / 2, method,
            ha="center", va="center", fontsize=8,
            fontweight="bold" if is_ours else "normal",
            color="#1A5276" if is_ours else "#2C3E50", linespacing=1.3)

    # Feature 셀
    for k, (flag, cw, x0) in enumerate(zip(flags, col_widths[2:], xs[2:])):
        bg = CELL_BG[flag] if not is_ours or flag == "Y" else CELL_BG[flag]
        # "Ours" + "Y" 셀은 더 진한 초록
        if is_ours and flag == "Y":
            bg = "#A9DFBF"
        ax.add_patch(plt.Rectangle((x0, y0), cw, row_h,
                                    facecolor=bg, edgecolor="#D5D8DC", linewidth=0.5))
        ax.text(x0 + cw / 2, y0 + row_h / 2, CELL_TEXT[flag],
                ha="center", va="center",
                fontsize=13 if flag != "P" else 11,
                fontweight="bold",
                color=CELL_FG[flag])

    # "Ours" 행 왼쪽 강조 선
    if is_ours:
        ax.add_patch(plt.Rectangle((0, y0), 0.08, row_h,
                                    facecolor="#E74C3C", edgecolor="none"))

# ── 테두리 ────────────────────────────────────────────────────────
ax.add_patch(plt.Rectangle((0, 0), total_w, n_rows * row_h + header_h,
                             fill=False, edgecolor="#2C3E50", linewidth=1.5))

# ── 제목 & 주석 ──────────────────────────────────────────────────
ax.set_title(
    "TABLE.  Comparison of Regime-Aware Portfolio Optimization Approaches",
    fontsize=11, fontweight="bold", pad=10, y=1.0,
)
refs = (
    "[1] Kim et al. (2019) JRFM  "
    "[2] Jiang et al. (2017) arXiv  "
    "[3] Chen et al. (2021) ICAIF  "
    "[4] Zhang et al. (2025) RegimeFolio arXiv:2510.14986"
)
fig.text(0.5, 0.01, refs, ha="center", fontsize=7.5, color="#555555", style="italic")

legend = [
    mpatches.Patch(facecolor="#A9DFBF", edgecolor="#1E8449", label="O  Present"),
    mpatches.Patch(facecolor="#FADBD8", edgecolor="#C0392B", label="X  Absent"),
    mpatches.Patch(facecolor="#FEF9E7", edgecolor="#B7770D", label="~  Implicit / Partial"),
]
fig.legend(handles=legend, loc="lower right", fontsize=8.5,
           bbox_to_anchor=(0.98, 0.04), framealpha=0.9, ncol=3)

plt.tight_layout(rect=[0, 0.06, 1, 1])
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/fig9_related_work.png", bbox_inches="tight")
plt.close()
print("저장: outputs/figures/fig9_related_work.png")

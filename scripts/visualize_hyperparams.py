"""
Fig 03. HMM 설정 + Conv1D+LSTM 하이퍼파라미터 표
RegimeFolio Table III/IV 스타일
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

HEADER_COLOR = "#1B2631"
OUR_COLOR    = "#EBF5FB"
ALT_COLOR    = "#F8F9FA"

def draw_table(ax, title, col_headers, rows, col_widths, highlight_rows=None):
    ax.axis("off")
    n_cols = len(col_headers)
    n_rows = len(rows)
    total_w = sum(col_widths)
    xs = [sum(col_widths[:i]) for i in range(n_cols)]
    row_h = 0.13
    header_h = 0.16
    total_h = header_h + n_rows * row_h + 0.05

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)

    # 헤더
    y_header = total_h - header_h
    for j, (hdr, cw, x0) in enumerate(zip(col_headers, col_widths, xs)):
        ax.add_patch(plt.Rectangle((x0, y_header), cw, header_h,
                                   facecolor=HEADER_COLOR, edgecolor="white", lw=0.5))
        ax.text(x0 + cw/2, y_header + header_h/2, hdr,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

    # 데이터 행
    for i, row in enumerate(rows):
        y0 = total_h - header_h - (i + 1) * row_h
        bold_row = highlight_rows and i in highlight_rows
        bg = OUR_COLOR if bold_row else (ALT_COLOR if i % 2 == 0 else "white")
        for j, (val, cw, x0) in enumerate(zip(row, col_widths, xs)):
            ax.add_patch(plt.Rectangle((x0, y0), cw, row_h,
                                       facecolor=bg, edgecolor="#D5D8DC", lw=0.4))
            ax.text(x0 + cw/2, y0 + row_h/2, str(val),
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if bold_row else "normal",
                    color="#1A5276" if bold_row else "#2C3E50")

    # 외곽선
    ax.add_patch(plt.Rectangle((0, total_h - header_h - n_rows * row_h),
                                total_w, header_h + n_rows * row_h,
                                fill=False, edgecolor="#2C3E50", lw=1.2))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)


fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("TABLE.  Model Configuration and Hyperparameter Settings",
             fontsize=12, fontweight="bold", y=0.97)

# ── 왼쪽: HMM 국면 레이블링 설정 ─────────────────────────────────
hmm_headers = ["Parameter", "Value", "Description"]
hmm_rows = [
    ["Input Feature",      "SPY Adj Close",          "S&P 500 ETF 일별 수익률"],
    ["Rolling Window",     "504 days (~2 years)",    "비정상성 대응을 위한 rolling HMM"],
    ["Number of States",   "3",                      "Bear / Neutral / Bull"],
    ["Emission Model",     "Gaussian",               "수익률 분포 가정"],
    ["Label Mapping",      "Sharpe-based",           "Low Sharpe→Bear, High Sharpe→Bull"],
    ["Lookahead",          "None",                   "미래 데이터 누수 없음"],
    ["Train/Val/Test",     "70% / 15% / 15%",        "시계열 순서 유지 분할"],
    ["Total Samples",      "698",                    "5일 단위, 30일 윈도우"],
]
draw_table(ax_left,
           "TABLE A.  HMM Regime Labeling Configuration",
           hmm_headers, hmm_rows,
           col_widths=[1.3, 1.6, 2.4])

# ── 오른쪽: Conv1D+LSTM 하이퍼파라미터 ───────────────────────────
hp_headers = ["Hyperparameter", "Value", "Search Range"]
hp_rows = [
    ["Input Shape",       "(30, 40)",          "30-day × 40-feature"],
    ["Conv Channels",     "16",                "[8, 16, 32]"],
    ["LSTM Hidden",       "32",                "[16, 32, 64]"],
    ["Dropout",           "0.6",               "[0.3, 0.5, 0.6]"],
    ["Optimizer",         "AdamW",             "Adam / AdamW"],
    ["Learning Rate",     "1e-4",              "[1e-4, 5e-4, 1e-3]"],
    ["Weight Decay",      "1e-2",              "[1e-3, 1e-2]"],
    ["Neutral Boost",     "1.2",               "[1.0, 1.2, 1.5]"],
    ["Batch Size",        "16",                "[16, 32]"],
    ["Max Epochs",        "80",                "Fixed"],
    ["Early Stopping",    "patience=10",       "val_balanced_accuracy"],
    ["Random Seed",       "42",                "Fixed"],
]
draw_table(ax_right,
           "TABLE B.  Conv1D+LSTM Hyperparameter Configuration",
           hp_headers, hp_rows,
           col_widths=[1.5, 1.2, 2.0])

plt.tight_layout(rect=[0, 0, 1, 0.94])
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/fig03_hyperparams.png", bbox_inches="tight")
plt.close()
print("저장: outputs/figures/fig03_hyperparams.png")

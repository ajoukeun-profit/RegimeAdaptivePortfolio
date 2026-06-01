"""
Fig 01. 발표용 파이프라인 다이어그램

핵심 흐름:
Raw ETF data -> HMM regime labels -> supervised dataset ->
Conv1D+LSTM regime probabilities -> Regime-MVO portfolio.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.dpi": 140,
})

# ── 단계별 내용 ───────────────────────────────────────────────────
PHASES = [
    {
        "title": "1. Market Data",
        "subtitle": "가격 데이터 수집",
        "items": ["SPY / QQQ / GLD / TLT", "일별 OHLCV", "기술지표 피처 계산"],
        "color": "#1A5276",
    },
    {
        "title": "2. HMM Labeling",
        "subtitle": "시장 국면 정의",
        "items": ["504일 rolling window", "3-state Gaussian HMM", "Bear / Neutral / Bull 매핑"],
        "color": "#1A5276",
    },
    {
        "title": "3. Supervised Dataset",
        "subtitle": "예측 문제 구성",
        "items": ["입력: 30일 × 40피처", "타겟: 5일 후 SPY 국면", "시간순 Train / Valid / Test"],
        "color": "#1A5276",
    },
    {
        "title": "4. Regime Classifier",
        "subtitle": "Conv1D + LSTM",
        "items": ["Conv1D: 단기 패턴", "LSTM: 30일 흐름", "출력: 국면 확률"],
        "color": "#1A5276",
    },
    {
        "title": "5. Regime-MVO",
        "subtitle": "확률 기반 자산배분",
        "items": ["국면별 MVO 비중 계산", "예측 확률로 가중평균", "Test MDD -7.2%"],
        "color": "#1A5276",
    },
]

# ── 레이아웃 계산 ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 7.6))
ax.set_xlim(0, 20)
ax.set_ylim(0, 7.6)
ax.axis("off")

BOX_W    = 3.35
BOX_H    = 5.25
HEADER_H = 1.35
GAP      = 0.55
START_X  = 0.35
Y_BOTTOM = 0.95

for i, phase in enumerate(PHASES):
    x0 = START_X + i * (BOX_W + GAP)
    y0 = Y_BOTTOM

    # ── 본문 박스 ────────────────────────────────────────────────
    body = FancyBboxPatch(
        (x0, y0), BOX_W, BOX_H,
        boxstyle="round,pad=0.05",
        facecolor="white", edgecolor=phase["color"], linewidth=1.8,
    )
    ax.add_patch(body)

    # ── 헤더 박스 ────────────────────────────────────────────────
    header = FancyBboxPatch(
        (x0, y0 + BOX_H - HEADER_H), BOX_W, HEADER_H,
        boxstyle="round,pad=0.05",
        facecolor=phase["color"], edgecolor=phase["color"], linewidth=1.8,
    )
    ax.add_patch(header)

    # ── 헤더 텍스트 ──────────────────────────────────────────────
    ax.text(x0 + BOX_W / 2, y0 + BOX_H - HEADER_H / 2 + 0.14,
            phase["title"],
            ha="center", va="center",
            fontsize=17, fontweight="bold", color="white")
    ax.text(x0 + BOX_W / 2, y0 + BOX_H - HEADER_H / 2 - 0.24,
            phase["subtitle"],
            ha="center", va="center",
            fontsize=14, color="#D6EAF8", linespacing=1.25)

    # ── 본문 항목 ────────────────────────────────────────────────
    item_y = y0 + BOX_H - HEADER_H - 0.65
    for item in phase["items"]:
        ax.text(x0 + 0.22, item_y, "•", ha="left", va="center",
                fontsize=20, color=phase["color"], fontweight="bold")
        ax.text(x0 + 0.58, item_y, item, ha="left", va="center",
                fontsize=15.5, color="#2C3E50", fontweight="bold", wrap=True)
        item_y -= 1.02

    # ── 화살표 (마지막 박스 제외) ─────────────────────────────────
    if i < len(PHASES) - 1:
        arrow_x = x0 + BOX_W + 0.04
        arrow_y = y0 + BOX_H / 2
        ax.annotate(
            "", xy=(arrow_x + GAP - 0.08, arrow_y),
            xytext=(arrow_x, arrow_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#1A5276",
                lw=3.2,
                mutation_scale=30,
            ),
        )

# ── 아래 흐름 레이블 ──────────────────────────────────────────────
labels = ["Raw data", "Regime labels", "Training data", "Predicted probabilities", "Portfolio weights"]
for i, label in enumerate(labels):
    x0 = START_X + i * (BOX_W + GAP)
    ax.text(x0 + BOX_W / 2, Y_BOTTOM - 0.28, label,
            ha="center", va="center", fontsize=13,
            color="#7F8C8D", style="italic")

ax.set_title(
    "Fig 01. Regime-Aware Portfolio Pipeline\n"
    "시장 데이터 → HMM 국면 라벨 → Conv1D+LSTM 국면 확률 → Regime-MVO 비중",
    fontsize=21, fontweight="bold", pad=14, y=1.01,
)

plt.tight_layout()
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/figures/final").mkdir(parents=True, exist_ok=True)
Path("outputs/figures/legacy").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/final/fig01_pipeline.png", bbox_inches="tight")
plt.savefig("outputs/figures/legacy/fig10_pipeline.png", bbox_inches="tight")
plt.close()
print("저장: outputs/figures/final/fig01_pipeline.png")

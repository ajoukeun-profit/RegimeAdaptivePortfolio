"""
발표용 시각화: 4개 그림 생성
  Fig1. 실험별 분류 성능 비교 (bar chart)
  Fig2. 백테스트 누적 수익률 곡선
  Fig3. 전략별 Sharpe / MDD / Calmar 비교
  Fig4. Exp3 Confusion Matrix
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn

# ── 공통 스타일 ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})
COLORS = {
    "bear":    "#E74C3C",
    "neutral": "#F39C12",
    "bull":    "#27AE60",
    "model":   "#2980B9",
    "gray":    "#95A5A6",
}


# ══════════════════════════════════════════════════════════════════
# Fig 1. 실험별 분류 성능 비교
# ══════════════════════════════════════════════════════════════════
exp_names  = ["Exp1\nBaseline", "Exp2\nFocal Loss", "Exp3\nAugmentation\n(최종)", "Exp4\nBiLSTM+Attn"]
accuracy   = [57.1, 52.4, 61.0, 49.5]
bear_acc   = [34.9, 34.9, 46.5, 30.2]
neutral_acc= [23.8, 47.6, 33.3, 38.1]
bull_acc   = [97.6, 73.2, 90.2, 75.6]

x     = np.arange(len(exp_names))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - 1.5*width, accuracy,    width, label="Overall Accuracy", color=COLORS["model"],   alpha=0.9)
b2 = ax.bar(x - 0.5*width, bear_acc,    width, label="Bear Recall",      color=COLORS["bear"],    alpha=0.8)
b3 = ax.bar(x + 0.5*width, neutral_acc, width, label="Neutral Recall",   color=COLORS["neutral"], alpha=0.8)
b4 = ax.bar(x + 1.5*width, bull_acc,    width, label="Bull Recall",      color=COLORS["bull"],    alpha=0.8)

# 최고값 표시
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Experiment Comparison: Classification Performance", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 115)
ax.legend(loc="upper right", fontsize=9)
ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random (33.3%)")
ax.text(3.7, 34.5, "Random\n33.3%", fontsize=7, color="gray")

# Exp3 강조
ax.axvspan(1.6, 2.4, alpha=0.06, color=COLORS["model"])

plt.tight_layout()
plt.savefig("outputs/figures/fig1_experiment_comparison.png", bbox_inches="tight")
plt.close()
print("Fig1 저장")


# ══════════════════════════════════════════════════════════════════
# Fig 2. 백테스트 누적 수익률 곡선
# ══════════════════════════════════════════════════════════════════

# 데이터 재계산
data      = np.load("data/processed/spy_supervised_30d_5d.npz", allow_pickle=True)
X_test    = data["X_test"].astype(np.float32)
index_rows = list(csv.DictReader(open("data/processed/spy_supervised_30d_5d_index.csv")))
test_rows  = [r for r in index_rows if r["split"] == "test"]

spy_prices = {}
with open("data/raw/spy_daily.csv") as f:
    for row in csv.DictReader(f):
        spy_prices[row["Date"]] = float(row["Adj Close"])
spy_dates = sorted(spy_prices.keys())

holding_returns = np.array([
    spy_prices[r["target_date"]] / spy_prices[r["input_end_date"]] - 1
    for r in test_rows
])
dates_axis = [r["target_date"] for r in test_rows]

# 모델 예측 (Exp3)
device = torch.device("cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__()
        pad = ks // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, ks, padding=pad), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, out_ch, ks, padding=pad), nn.BatchNorm1d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.net(x.transpose(1,2)).transpose(1,2)

class RegimeClassifier_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(10, 32)
        self.lstm = nn.LSTM(32, 64, 1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 3)
        )
    def forward(self, x):
        x = self.conv(x); _, (h, _) = self.lstm(x); return self.classifier(h[-1])
    def predict_proba(self, x): return torch.softmax(self.forward(x), dim=-1)

model = RegimeClassifier_V1()
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    probs   = model.predict_proba(torch.tensor(X_test)).numpy()
w_model = probs[:, 2] + 0.5 * probs[:, 1]

def cumulative_curve(weights, holding_rets, cost=0.001):
    w_prev   = np.concatenate([[0.0], weights[:-1]])
    port_ret = weights * holding_rets - np.abs(weights - w_prev) * cost
    return np.cumprod(1 + port_ret)

strategies = {
    "Buy & Hold":          np.ones(105),
    "60/40":               np.full(105, 0.6),
    "Conv1D+LSTM (ours)":  w_model,
}

# MA Crossover
def get_sma(end_date, window):
    idx = spy_dates.index(end_date) if end_date in spy_dates else -1
    if idx < window: return spy_prices[end_date]
    return np.mean([spy_prices[d] for d in spy_dates[idx-window+1:idx+1]])

ma_weights = [1.0 if get_sma(r["input_end_date"], 20) > get_sma(r["input_end_date"], 60)
              else 0.0 for r in test_rows]
strategies["MA Crossover"] = np.array(ma_weights)

style = {
    "Buy & Hold":         ("--", "#2C3E50", 1.5),
    "60/40":              ("--", COLORS["gray"], 1.2),
    "MA Crossover":       (":",  "#8E44AD", 1.2),
    "Conv1D+LSTM (ours)": ("-",  COLORS["model"], 2.5),
}

fig, ax = plt.subplots(figsize=(12, 5))
for name, weights in strategies.items():
    curve = np.concatenate([[1.0], cumulative_curve(weights, holding_returns)])
    ls, color, lw = style[name]
    ax.plot(range(len(curve)), (curve - 1) * 100,
            linestyle=ls, color=color, linewidth=lw, label=name)

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Rebalancing Period (5-day intervals)")
ax.set_ylabel("Cumulative Return (%)")
ax.set_title("Backtest: Cumulative Return (2024.04 ~ 2026.05)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)

# x축 날짜 레이블 (20개 간격)
tick_idx = list(range(0, len(dates_axis)+1, 20))
tick_lbl = ["Start"] + [dates_axis[i-1][:7] for i in tick_idx[1:] if i <= len(dates_axis)]
ax.set_xticks(tick_idx[:len(tick_lbl)])
ax.set_xticklabels(tick_lbl, rotation=30, fontsize=8)

plt.tight_layout()
plt.savefig("outputs/figures/fig2_cumulative_return.png", bbox_inches="tight")
plt.close()
print("Fig2 저장")


# ══════════════════════════════════════════════════════════════════
# Fig 3. 전략별 위험-수익 비교 (Sharpe, MDD, Calmar)
# ══════════════════════════════════════════════════════════════════
with open("outputs/results/backtest_results.json") as f:
    bt = json.load(f)

strat_order = ["Buy & Hold", "80/20", "60/40", "MA Crossover",
               "Vol Targeting", "40/60", "Conv1D+LSTM (ours)"]
sharpes  = [bt[s]["sharpe"]        for s in strat_order]
mdds     = [abs(bt[s]["mdd"]) * 100 for s in strat_order]
calmars  = [bt[s]["calmar"]        for s in strat_order]
bar_colors = [COLORS["model"] if "LSTM" in s else COLORS["gray"] for s in strat_order]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Strategy Comparison: Risk-Adjusted Metrics", fontsize=13, fontweight="bold")

short_names = [s.replace("Conv1D+LSTM (ours)", "Ours ◀") for s in strat_order]

for ax, values, title, higher_better in zip(
    axes,
    [sharpes, mdds, calmars],
    ["Sharpe Ratio\n(higher = better)", "Max Drawdown (%)\n(lower = better)", "Calmar Ratio\n(higher = better)"],
    [True, False, True]
):
    bars = ax.barh(short_names, values, color=bar_colors, alpha=0.85)
    best_val = min(values) if not higher_better else max(values)
    best_idx = values.index(best_val)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)

    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("outputs/figures/fig3_strategy_metrics.png", bbox_inches="tight")
plt.close()
print("Fig3 저장")


# ══════════════════════════════════════════════════════════════════
# Fig 4. Exp3 Confusion Matrix
# ══════════════════════════════════════════════════════════════════
cm = np.array([[20, 5, 18],
               [ 8, 7,  6],
               [ 1, 3, 37]])
labels = ["Bear\n(0)", "Neutral\n(1)", "Bull\n(2)"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f"Pred\n{l}" for l in labels], fontsize=10)
ax.set_yticklabels([f"True\n{l}" for l in labels], fontsize=10)

for i in range(3):
    for j in range(3):
        color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)

total = [43, 21, 41]
recall = [cm[i,i]/total[i]*100 for i in range(3)]
for i, (r, name) in enumerate(zip(recall, ["Bear", "Neutral", "Bull"])):
    ax.text(3.1, i, f"Recall\n{r:.1f}%", va="center", fontsize=9,
            color=COLORS[name.lower()])

ax.set_title("Confusion Matrix — Exp3 (Best Model)\nTest set (105 samples)", fontsize=11, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("outputs/figures/fig4_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("Fig4 저장")

print("\n모든 그림 저장 완료: outputs/figures/fig*.png")

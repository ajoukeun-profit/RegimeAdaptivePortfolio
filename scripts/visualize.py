"""
발표용 시각화: 4개 그림 생성
  Fig1. 3단계 실험별 분류 성능 비교 (Phase 1~3)
  Fig2. 백테스트 누적 수익률 곡선  (cross-asset 데이터 + 현재 모델)
  Fig3. 전략별 Sharpe / MDD / Calmar 비교
  Fig4. 최종 모델 Confusion Matrix
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier

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

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# ══════════════════════════════════════════════════════════════════
# Fig 1. 3단계 실험별 분류 성능 비교
# ══════════════════════════════════════════════════════════════════
# Phase 1: SPY 단독 (대표 2개)
# Phase 2: 4자산 각자 라벨 (대표 1개)
# Phase 3: Cross-asset + AdamW + Neutral-boost (최종)
exp_names  = [
    "Ph1\nBaseline",
    "Ph1\nAugment",
    "Ph2\nMulti-label",
    "Ph3\nCross-asset\n(최종)",
]
accuracy   = [57.1, 61.0, 59.8, 61.9]
bear_acc   = [34.9, 46.5, 58.8, 60.5]
neutral_acc= [23.8, 33.3, 25.3,  0.0]
bull_acc   = [97.6, 90.2, 80.6, 95.1]

x     = np.arange(len(exp_names))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - 1.5*width, accuracy,    width, label="Overall Accuracy", color=COLORS["model"],   alpha=0.9)
b2 = ax.bar(x - 0.5*width, bear_acc,    width, label="Bear Recall",      color=COLORS["bear"],    alpha=0.8)
b3 = ax.bar(x + 0.5*width, neutral_acc, width, label="Neutral Recall",   color=COLORS["neutral"], alpha=0.8)
b4 = ax.bar(x + 1.5*width, bull_acc,    width, label="Bull Recall",      color=COLORS["bull"],    alpha=0.8)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Phase별 분류 성능 비교 (대표 실험)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 115)
ax.legend(loc="upper right", fontsize=9)
ax.axhline(33.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(3.7, 34.5, "Random\n33.3%", fontsize=7, color="gray")

# Phase 3 강조
ax.axvspan(2.6, 3.4, alpha=0.06, color=COLORS["model"])

plt.tight_layout()
plt.savefig("outputs/figures/fig1_experiment_comparison.png", bbox_inches="tight")
plt.close()
print("Fig1 저장")


# ══════════════════════════════════════════════════════════════════
# Fig 2. 백테스트 누적 수익률 곡선  (cross-asset 데이터 기준)
# ══════════════════════════════════════════════════════════════════
data       = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
X_test     = torch.tensor(data["X_test"].astype(np.float32)).to(device)
index_rows = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
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
n = len(test_rows)

# 모델 예측 (cross-asset, input_size=40)
model = RegimeClassifier(input_size=40, conv_channels=16, lstm_hidden=32).to(device)
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
model.eval()
with torch.no_grad():
    probs = model.predict_proba(X_test).cpu().numpy()
w_model = probs[:, 2] + 0.5 * probs[:, 1]

def cumulative_curve(weights, holding_rets, cost=0.001):
    w_prev   = np.concatenate([[0.0], weights[:-1]])
    port_ret = weights * holding_rets - np.abs(weights - w_prev) * cost
    return np.cumprod(1 + port_ret)

def get_sma(end_date, window):
    idx = spy_dates.index(end_date) if end_date in spy_dates else -1
    if idx < window:
        return spy_prices[end_date]
    return np.mean([spy_prices[d] for d in spy_dates[idx - window + 1:idx + 1]])

ma_weights = [
    1.0 if get_sma(r["input_end_date"], 20) > get_sma(r["input_end_date"], 60) else 0.0
    for r in test_rows
]

# Regime Momentum Tilt 수익률 계산 (weights CSV 사용)
qqq_prices = {}
gld_prices = {}
tlt_prices = {}
for fname, d in [("data/raw/qqq_daily.csv", qqq_prices),
                 ("data/raw/gld_daily.csv",  gld_prices),
                 ("data/raw/tlt_daily.csv",  tlt_prices)]:
    with open(fname) as f:
        for row in csv.DictReader(f):
            d[row["Date"]] = float(row["Adj Close"])

wt_rows = list(csv.DictReader(open("outputs/results/backtest_regime_momentum_weights.csv")))
rmt_weights = []
for row in wt_rows:
    rmt_weights.append({
        "SPY": float(row["SPY"]), "QQQ": float(row["QQQ"]),
        "GLD": float(row["GLD"]), "TLT": float(row["TLT"]),
        "CASH": float(row["CASH"]),
    })

def get_price(d, date):
    return d.get(date, None)

def rmt_curve(wt_rows_list, rmt_w, spy_p, qqq_p, gld_p, tlt_p, cost=0.001):
    cum = [1.0]
    prev_w = None
    asset_prices = {"SPY": spy_p, "QQQ": qqq_p, "GLD": gld_p, "TLT": tlt_p}
    for i, (row, w) in enumerate(zip(wt_rows_list[:-1], rmt_w[:-1])):
        d0 = row["date"]
        d1 = wt_rows_list[i + 1]["date"]
        ret = 0.0
        for asset, p in asset_prices.items():
            p0, p1 = get_price(p, d0), get_price(p, d1)
            if p0 and p1 and p0 > 0:
                ret += w[asset] * (p1 / p0 - 1)
        tc = 0.0
        if prev_w:
            tc = sum(abs(w[a] - prev_w[a]) for a in asset_prices) * cost
        cum.append(cum[-1] * (1 + ret - tc))
        prev_w = w
    return np.array(cum)

rmt_cum = rmt_curve(wt_rows, rmt_weights, spy_prices, qqq_prices, gld_prices, tlt_prices)

# EW 수익률 계산 (4자산 1/N: SPY/QQQ/GLD/TLT)
def load_prices_vis(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row["Date"]] = float(row["Adj Close"])
    return d

qqq_prices_v = load_prices_vis("data/raw/qqq_daily.csv")
gld_prices_v = load_prices_vis("data/raw/gld_daily.csv")
tlt_prices_v = load_prices_vis("data/raw/tlt_daily.csv")

def asset_ret(prices, d0, d1):
    return prices[d1] / prices[d0] - 1.0 if d0 in prices and d1 in prices else 0.0

ew_holding = np.array([
    0.25 * asset_ret(spy_prices, r["input_end_date"], r["target_date"])
    + 0.25 * asset_ret(qqq_prices_v, r["input_end_date"], r["target_date"])
    + 0.25 * asset_ret(gld_prices_v, r["input_end_date"], r["target_date"])
    + 0.25 * asset_ret(tlt_prices_v, r["input_end_date"], r["target_date"])
    for r in test_rows
])
ew_curve = np.concatenate([[1.0], np.cumprod(1 + ew_holding)])

strategies = {
    "Buy & Hold":   np.ones(n),
    "60/40":        np.full(n, 0.6),
    "MA Crossover": np.array(ma_weights),
}
style = {
    "Buy & Hold":                    ("--", "#2C3E50", 1.5),
    "60/40":                         ("--", COLORS["gray"], 1.2),
    "MA Crossover":                  (":",  "#8E44AD", 1.2),
    "EW 1/N (논문 벤치마크)":         ("--", "#27AE60", 1.5),
    "Conv1D+LSTM (SPY/Cash)":        ("-",  "#2980B9", 2.0),
    "Regime Momentum Tilt (ours)":   ("-",  "#E67E22", 2.5),
}

# Conv1D+LSTM SPY/Cash 누적 수익률: p_bull + 0.5*p_neutral → SPY 비중, 나머지 현금
lstm_spy_cash_curve = np.concatenate([[1.0], cumulative_curve(w_model, holding_returns)])

fig, ax = plt.subplots(figsize=(12, 5))
for name, weights in strategies.items():
    curve = np.concatenate([[1.0], cumulative_curve(weights, holding_returns)])
    ls, color, lw = style[name]
    ax.plot(range(len(curve)), (curve - 1) * 100,
            linestyle=ls, color=color, linewidth=lw, label=name)
# EW
ax.plot(range(len(ew_curve)), (ew_curve - 1) * 100,
        linestyle="--", color="#27AE60", linewidth=1.5, label="EW 1/N (논문 벤치마크)")
# Conv1D+LSTM SPY/Cash
ls, color, lw = style["Conv1D+LSTM (SPY/Cash)"]
ax.plot(range(len(lstm_spy_cash_curve)), (lstm_spy_cash_curve - 1) * 100,
        linestyle=ls, color=color, linewidth=lw, label="Conv1D+LSTM (SPY/Cash)")
# Regime Momentum Tilt
ls, color, lw = style["Regime Momentum Tilt (ours)"]
ax.plot(range(len(rmt_cum)), (rmt_cum - 1) * 100,
        linestyle=ls, color=color, linewidth=lw, label="Regime Momentum Tilt (ours)")

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Rebalancing Period (5-day intervals)")
ax.set_ylabel("Cumulative Return (%)")
ax.set_title("Backtest: Cumulative Return (2024.04 ~ 2026.05)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)

tick_idx = list(range(0, len(dates_axis) + 1, 20))
tick_lbl = ["Start"] + [dates_axis[i - 1][:7] for i in tick_idx[1:] if i <= len(dates_axis)]
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
with open("outputs/results/backtest_regime_momentum_results.json") as f:
    bt_rmt = json.load(f)

rmt_data = bt_rmt["Regime Momentum Tilt"]
strat_order = ["Buy & Hold", "EW (1/N)", "60/40", "80/20", "40/60",
               "MA Crossover", "Vol Targeting", "Regime Momentum Tilt (ours)"]
all_data = {**bt, "Regime Momentum Tilt (ours)": rmt_data}
sharpes  = [all_data[s]["sharpe"]         for s in strat_order]
mdds     = [abs(all_data[s]["mdd"]) * 100 for s in strat_order]
calmars  = [all_data[s]["calmar"]         for s in strat_order]
bar_colors = ["#E67E22" if "Regime" in s else "#27AE60" if "EW" in s else COLORS["gray"] for s in strat_order]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Strategy Comparison: Risk-Adjusted Metrics", fontsize=13, fontweight="bold")

short_names = [s.replace("Regime Momentum Tilt (ours)", "Regime Tilt ◀").replace("EW (1/N)", "EW 1/N") for s in strat_order]

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
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/figures/fig3_strategy_metrics.png", bbox_inches="tight")
plt.close()
print("Fig3 저장")


# ══════════════════════════════════════════════════════════════════
# Fig 4. 최종 모델 Confusion Matrix (best_model.pt 실제 결과)
# ══════════════════════════════════════════════════════════════════
# 실제 test 결과: Bear(43): 26/0/17, Neutral(21): 8/0/13, Bull(41): 2/0/39
cm     = np.array([[26, 0, 17],
                   [ 8, 0, 13],
                   [ 2, 0, 39]])
labels = ["Bear\n(0)", "Neutral\n(1)", "Bull\n(2)"]
totals = [43, 21, 41]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f"Pred\n{l}" for l in labels], fontsize=10)
ax.set_yticklabels([f"True\n{l}" for l in labels], fontsize=10)

for i in range(3):
    for j in range(3):
        color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)

recall = [cm[i, i] / totals[i] * 100 for i in range(3)]
for i, (r, name) in enumerate(zip(recall, ["Bear", "Neutral", "Bull"])):
    ax.text(3.1, i, f"Recall\n{r:.1f}%", va="center", fontsize=9,
            color=COLORS[name.lower()])

ax.set_title(
    "Confusion Matrix — 최종 모델 (Phase 3)\n"
    "Cross-asset + AdamW + neutral-boost 1.2, seed=42  |  Test set (105 samples)",
    fontsize=10, fontweight="bold", pad=12
)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("outputs/figures/fig4_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("Fig4 저장")

print("\n모든 그림 저장 완료: outputs/figures/fig*.png")

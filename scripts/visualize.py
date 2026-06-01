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
# Fig 1. 논문 스타일 분류 성능 비교 표
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.axis("off")

col_labels = ["Experiment", "Architecture", "Accuracy", "Bear\nRecall", "Neutral\nRecall", "Bull\nRecall"]
row_data = [
    ["Ph1  Baseline",            "Conv1D+LSTM  (SPY, 10 features)",              "57.1%", "34.9%", "23.8%", "97.6%"],
    ["Ph1  Augmentation",        "Conv1D+LSTM  (SPY, augmented)",                "61.0%", "46.5%", "33.3%", "90.2%"],
    ["Ph2  Multi-asset labels",  "Conv1D+LSTM  (4-asset ind. labels)",           "59.8%", "58.8%", "25.3%", "80.6%"],
    ["Ph3  Cross-asset + AdamW ★","Conv1D+LSTM  (4-asset shared label, final)",  "61.9%", "60.5%",  "0.0%", "95.1%"],
]

table = ax.table(
    cellText=row_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0.18, 1, 0.78],
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)

# 헤더 스타일
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_height(0.28)

# 행 스타일
for i in range(1, 5):
    for j in range(len(col_labels)):
        cell = table[i, j]
        if i == 4:  # 최종 행 강조
            cell.set_facecolor("#EBF5FB")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FAFAFA" if i % 2 == 0 else "white")
        # Bear Recall 열 (j=3) 색상
        if j == 3 and i > 0:
            cell.set_text_props(color="#E74C3C", fontweight="bold")
        cell.set_height(0.22)

table.auto_set_column_width([0, 1, 2, 3, 4, 5])

ax.set_title(
    "Table 1.  Classification Performance by Experimental Phase",
    fontsize=12, fontweight="bold", pad=6, y=0.98
)

# 메트릭 설명
notes = (
    "† Accuracy: overall correct / total samples.   "
    "Recall (per class) = TP / all actual positives in that class  "
    "(higher = better for Bear).   Random baseline: 33.3% per class.\n"
    "Bear Recall is the primary optimization target — early detection of downturns drives portfolio protection."
)
fig.text(0.5, 0.04, notes, ha="center", fontsize=8, style="italic", color="#555555",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#BDC3C7", alpha=0.8))

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
    "DL Regime SPY/Cash":            ("-",  "#2980B9", 2.0),
    "Regime Momentum Tilt (ours)":   ("-",  "#E67E22", 2.5),
}

# DL Regime SPY/Cash 누적 수익률: p_bull + 0.5*p_neutral → SPY 비중, 나머지 현금
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
# DL Regime SPY/Cash
ls, color, lw = style["DL Regime SPY/Cash"]
ax.plot(range(len(lstm_spy_cash_curve)), (lstm_spy_cash_curve - 1) * 100,
        linestyle=ls, color=color, linewidth=lw, label="DL Regime SPY/Cash")
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
# Fig 3. 전략별 위험-수익 비교 (Sharpe, MDD, Calmar) — 최종 7개 전략
# ══════════════════════════════════════════════════════════════════
with open("outputs/results/backtest_mvo_results.json") as f:
    bt_mvo = json.load(f)
with open("outputs/results/backtest_results.json") as f:
    bt = json.load(f)
with open("outputs/results/backtest_regime_momentum_results.json") as f:
    bt_rmt = json.load(f)

all_data_fig3 = {
    "Buy & Hold":             bt_mvo["Buy & Hold"],
    "EW 1/N":                 bt_mvo["EW 1/N"],
    "60/40":                  bt_mvo["60/40"],
    "MA Crossover":           bt["MA Crossover"],
    "DL Regime\nSPY/Cash":   bt_mvo["DL Regime SPY/Cash"],
    "Regime Tilt":            bt_rmt["Regime Momentum Tilt"],
    "Regime-MVO ★\n(ours)":   bt_mvo["Regime-MVO (ours)"],
}
strat_order3 = list(all_data_fig3.keys())
sharpes3  = [all_data_fig3[s]["sharpe"]         for s in strat_order3]
mdds3     = [abs(all_data_fig3[s]["mdd"]) * 100 for s in strat_order3]
calmars3  = [all_data_fig3[s]["calmar"]         for s in strat_order3]

def bar_color3(name):
    if "Regime-MVO" in name:  return "#E74C3C"
    if "Regime Tilt" in name: return "#E67E22"
    if "EW" in name:          return "#27AE60"
    if "DL Regime" in name:   return "#2980B9"
    return COLORS["gray"]

colors3 = [bar_color3(s) for s in strat_order3]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Strategy Comparison: Risk-Adjusted Metrics  (Test: 2024.04 ~ 2026.05)",
             fontsize=13, fontweight="bold")

for ax, values, title, higher_better in zip(
    axes,
    [sharpes3, mdds3, calmars3],
    ["Sharpe Ratio\n(higher = better)", "Max Drawdown (%)\n(lower = better)", "Calmar Ratio\n(higher = better)"],
    [True, False, True],
):
    bars = ax.barh(strat_order3, values, color=colors3, alpha=0.85)
    best_val = min(values) if not higher_better else max(values)
    best_idx = values.index(best_val)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold")

import matplotlib.patches as mpatches
leg_handles = [
    mpatches.Patch(color="#E74C3C", label="Regime-MVO (ours)"),
    mpatches.Patch(color="#E67E22", label="Regime Momentum Tilt"),
    mpatches.Patch(color="#2980B9", label="DL Regime SPY/Cash"),
    mpatches.Patch(color="#27AE60", label="EW 1/N"),
    mpatches.Patch(color=COLORS["gray"], label="기타 전략"),
]
fig.legend(handles=leg_handles, loc="lower center", ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 1])
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

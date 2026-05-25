"""
2022년 하락장 구간 별도 백테스트
validation set (2022-03 ~ 2022-12) 기간 분석
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})

device = torch.device("cpu")

# ── 모델 정의 ────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super().__init__()
        pad = ks // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, ks, padding=pad), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, out_ch, ks, padding=pad), nn.BatchNorm1d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.net(x.transpose(1,2)).transpose(1,2)

class RegimeClassifier(nn.Module):
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

model = RegimeClassifier()
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location="cpu"))
model.eval()

# ── 데이터 로드 ──────────────────────────────────────────────────
data  = np.load("data/processed/spy_supervised_30d_5d.npz", allow_pickle=True)
X_valid = data["X_valid"].astype(np.float32)   # (105, 30, 10)

index_rows = list(csv.DictReader(open("data/processed/spy_supervised_30d_5d_index.csv")))
valid_rows  = [r for r in index_rows if r["split"] == "valid"]   # 105개

spy_prices = {}
with open("data/raw/spy_daily.csv") as f:
    for row in csv.DictReader(f):
        spy_prices[row["Date"]] = float(row["Adj Close"])
spy_dates = sorted(spy_prices.keys())

# ── 2022 구간 필터링 ─────────────────────────────────────────────
# valid_rows와 X_valid는 같은 순서로 대응됨 (index 0~104)
idx_2022 = [
    i for i, r in enumerate(valid_rows)
    if r["target_date"].startswith("2022")
]
rows_2022  = [valid_rows[i] for i in idx_2022]
X_2022     = X_valid[idx_2022]   # (40, 30, 10)

print(f"2022 구간: {rows_2022[0]['input_end_date']} ~ {rows_2022[-1]['target_date']}")
print(f"샘플 수: {len(rows_2022)}개")

# ── 모델 예측 ────────────────────────────────────────────────────
with torch.no_grad():
    probs_2022 = model.predict_proba(torch.tensor(X_2022)).numpy()

w_model = probs_2022[:, 2] + 0.5 * probs_2022[:, 1]

# ── holding period 수익률 ────────────────────────────────────────
holding_returns = np.array([
    spy_prices[r["target_date"]] / spy_prices[r["input_end_date"]] - 1
    for r in rows_2022
])

# ── baseline 전략 ────────────────────────────────────────────────
def get_sma(end_date, window):
    idx = spy_dates.index(end_date) if end_date in spy_dates else -1
    if idx < window: return spy_prices[end_date]
    return np.mean([spy_prices[d] for d in spy_dates[idx-window+1:idx+1]])

n = len(rows_2022)
ma_weights = np.array([
    1.0 if get_sma(r["input_end_date"], 20) > get_sma(r["input_end_date"], 60) else 0.0
    for r in rows_2022
])

strategies = {
    "Buy & Hold":         np.ones(n),
    "60/40":              np.full(n, 0.6),
    "MA Crossover":       ma_weights,
    "Conv1D+LSTM (ours)": w_model,
}

# ── 성과 계산 ────────────────────────────────────────────────────
COST = 0.001
RF   = 0.05
PPY  = 252 / 5

def metrics(weights, rets):
    w_prev   = np.concatenate([[0.0], weights[:-1]])
    port_ret = weights * rets - np.abs(weights - w_prev) * COST
    cum      = np.prod(1 + port_ret) - 1
    ann_ret  = (1 + cum) ** (PPY / len(port_ret)) - 1
    ann_vol  = port_ret.std() * np.sqrt(PPY)
    sharpe   = (ann_ret - RF) / ann_vol if ann_vol > 0 else 0.0
    curve    = np.cumprod(1 + port_ret)
    mdd      = (curve / np.maximum.accumulate(curve) - 1).min()
    calmar   = ann_ret / abs(mdd) if mdd < 0 else 0.0
    return {"cum": cum, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "mdd": mdd, "calmar": calmar, "port_ret": port_ret}

results = {name: metrics(w, holding_returns) for name, w in strategies.items()}

# ── 결과 출력 ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"2022 하락장 백테스트: {rows_2022[0]['input_end_date']} ~ {rows_2022[-1]['target_date']}")
print(f"{'='*65}")
print(f"{'전략':<22} {'누적수익':>8} {'연수익':>7} {'변동성':>7} {'Sharpe':>7} {'MDD':>8}")
print("─" * 65)
for name, m in results.items():
    marker = " ◀" if "LSTM" in name else ""
    print(f"{name:<22} {m['cum']:>7.1%}  {m['ann_ret']:>6.1%}  "
          f"{m['ann_vol']:>6.1%}  {m['sharpe']:>6.2f}  {m['mdd']:>7.1%}{marker}")

# ── 시각화 ───────────────────────────────────────────────────────
COLORS = {"Buy & Hold": ("#2C3E50", "--", 1.5),
          "60/40":      ("#95A5A6", "--", 1.2),
          "MA Crossover": ("#8E44AD", ":",  1.2),
          "Conv1D+LSTM (ours)": ("#2980B9", "-", 2.5)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("2022 Bear Market Backtest\n(SPY -18.6%, 2022.03 ~ 2022.12)",
             fontsize=13, fontweight="bold")

# 누적 수익률 곡선
dates_axis = [r["target_date"] for r in rows_2022]
for name, m in results.items():
    curve = np.concatenate([[1.0], np.cumprod(1 + m["port_ret"])])
    color, ls, lw = COLORS[name]
    ax1.plot(range(len(curve)), (curve - 1) * 100,
             color=color, linestyle=ls, linewidth=lw, label=name)

ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_ylabel("누적 수익률 (%)")
ax1.set_title("누적 수익률 비교")
tick_idx = list(range(0, len(dates_axis)+1, 8))
tick_lbl = ["Start"] + [dates_axis[min(i-1, len(dates_axis)-1)][:7] for i in tick_idx[1:]]
ax1.set_xticks(tick_idx[:len(tick_lbl)])
ax1.set_xticklabels(tick_lbl, rotation=30, fontsize=8)
ax1.legend(fontsize=9)

# MDD 비교 막대
names_short = ["Buy &\nHold", "60/40", "MA\nCrossover", "Ours\n◀"]
mdds = [abs(results[n]["mdd"]) * 100 for n in strategies]
bar_colors = ["#95A5A6", "#95A5A6", "#95A5A6", "#2980B9"]
bars = ax2.bar(names_short, mdds, color=bar_colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, mdds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("Max Drawdown (%)")
ax2.set_title("Max Drawdown 비교\n(낮을수록 좋음)")
bars[-1].set_edgecolor("gold")
bars[-1].set_linewidth(2.5)

plt.tight_layout()
plt.savefig("outputs/figures/fig5_2022_bear_backtest.png", bbox_inches="tight")
plt.close()
print("\n그래프 저장: outputs/figures/fig5_2022_bear_backtest.png")

# JSON 저장
save = {k: {kk: float(vv) for kk, vv in v.items() if kk != "port_ret"}
        for k, v in results.items()}
with open("outputs/results/backtest_2022_results.json", "w") as f:
    json.dump(save, f, indent=2)

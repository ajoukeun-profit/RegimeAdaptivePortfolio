"""
2022 Bear Market Backtest — Phase 3 Cross-asset 모델 기준
Validation set에서 2022 구간 (2022-03-11 ~ 2022-12-27, 40 samples) 추출

전략: Buy & Hold / 60/40 / MA Crossover / EW 1/N / Regime Momentum Tilt
출력: outputs/results/backtest_2022_results.json
      outputs/figures/fig5_2022_bear_backtest.png
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier
from regime_portfolio_policy import (
    ASSETS, ASSET_COLS,
    compute_return_seeking_weights,
    get_period_return,
)

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

COST = 0.001
RF   = 0.05
PPY  = 252 / 5   # 5거래일 단위


# ── 1. 데이터 로드 ────────────────────────────────────────────────
data     = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
X_train  = data["X_train"].astype(np.float32)
y_train  = data["y_train"]
X_valid  = data["X_valid"].astype(np.float32)

rows_all   = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
train_rows = [r for r in rows_all if r["split"] == "train"]
valid_rows = [r for r in rows_all if r["split"] == "valid"]


def load_prices(path: str) -> dict:
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row["Date"]] = float(row["Adj Close"])
    return d


spy_prices = load_prices("data/raw/spy_daily.csv")
qqq_prices = load_prices("data/raw/qqq_daily.csv")
gld_prices = load_prices("data/raw/gld_daily.csv")
tlt_prices = load_prices("data/raw/tlt_daily.csv")
spy_dates  = sorted(spy_prices.keys())

prices_by_asset = {"SPY": spy_prices, "QQQ": qqq_prices, "GLD": gld_prices, "TLT": tlt_prices}


# ── 2. 2022 구간 필터링 ───────────────────────────────────────────
idx_2022  = [i for i, r in enumerate(valid_rows) if r["target_date"].startswith("2022")]
rows_2022 = [valid_rows[i] for i in idx_2022]
X_2022    = X_valid[idx_2022]
n         = len(rows_2022)
print(f"2022 구간: {rows_2022[0]['input_end_date']} ~ {rows_2022[-1]['target_date']}  ({n}개)")


# ── 3. 모델 예측 ──────────────────────────────────────────────────
model = RegimeClassifier(input_size=40, conv_channels=16, lstm_hidden=32).to(device)
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    probs_2022 = model.predict_proba(torch.tensor(X_2022).to(device)).cpu().numpy()  # (n, 3)


# ── 4. 수익률 계산 ────────────────────────────────────────────────
spy_rets = np.array([
    spy_prices[r["target_date"]] / spy_prices[r["input_end_date"]] - 1
    for r in rows_2022
])

ew_rets = np.array([
    0.25 * sum(
        get_period_return(prices_by_asset[a], r["input_end_date"], r["target_date"])
        for a in ASSETS
    )
    for r in rows_2022
])


# ── 4-1. Regime-MVO 비중 계산 ────────────────────────────────────
def returns_matrix(rows):
    R = np.zeros((len(rows), len(ASSETS)))
    for i, row in enumerate(rows):
        for j, asset in enumerate(ASSETS):
            R[i, j] = get_period_return(
                prices_by_asset[asset],
                row["input_end_date"],
                row["target_date"],
            )
    return R


RF_P = RF / PPY


def max_sharpe_weights(R: np.ndarray, rf: float = RF_P) -> np.ndarray:
    n_assets = R.shape[1]
    w0 = np.ones(n_assets) / n_assets

    def neg_sharpe(w):
        port = R @ w
        mu = port.mean() - rf
        sig = port.std(ddof=0)
        return -(mu / sig) if sig > 1e-8 else 0.0

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return result.x if result.success else w0


train_R = returns_matrix(train_rows)
mvo_w = {rid: max_sharpe_weights(train_R[y_train == rid]) for rid in [0, 1, 2]}
regime_mvo_weights = (
    probs_2022[:, 0:1] * mvo_w[0]
    + probs_2022[:, 1:2] * mvo_w[1]
    + probs_2022[:, 2:3] * mvo_w[2]
)


# ── 5. Regime Momentum Tilt 비중 계산 ────────────────────────────
rmt_weights_list = []
prev_w = None
for p, row in zip(probs_2022, rows_2022):
    w = compute_return_seeking_weights(
        p, prices_by_asset, asof_date=row["input_end_date"], prev_w=prev_w
    )
    rmt_weights_list.append(w)
    prev_w = w
rmt_weights = np.array(rmt_weights_list)   # (n, 5)


# ── 6. 성과 계산 함수 ─────────────────────────────────────────────
def spy_metrics(weights: np.ndarray, rets: np.ndarray, name: str) -> dict:
    w_prev   = np.concatenate([[0.0], weights[:-1]])
    port_ret = weights * rets - np.abs(weights - w_prev) * COST
    cum      = float(np.prod(1 + port_ret) - 1)
    ann_ret  = float((1 + cum) ** (PPY / len(port_ret)) - 1)
    ann_vol  = float(port_ret.std() * np.sqrt(PPY))
    sharpe   = float((ann_ret - RF) / ann_vol) if ann_vol > 0 else 0.0
    curve    = np.cumprod(1 + port_ret)
    mdd      = float((curve / np.maximum.accumulate(curve) - 1).min())
    calmar   = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0
    return {"name": name, "cum": cum, "cum_ret": cum,
            "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "mdd": mdd, "calmar": calmar,
            "_port_ret": port_ret}


def rmt_metrics(weights: np.ndarray, name: str) -> dict:
    rets = []
    prev = np.zeros(weights.shape[1])
    for w, row in zip(weights, rows_2022):
        asset_rets = np.array(
            [get_period_return(prices_by_asset[a], row["input_end_date"], row["target_date"])
             for a in ASSETS] + [0.0]
        )
        turnover = float(np.sum(np.abs(w[:4] - prev[:4])))
        rets.append(float(w @ asset_rets - turnover * COST))
        prev = w
    rets = np.array(rets)
    cum     = float(np.prod(1 + rets) - 1)
    ann_ret = float((1 + cum) ** (PPY / len(rets)) - 1)
    ann_vol = float(rets.std() * np.sqrt(PPY))
    sharpe  = float((ann_ret - RF) / ann_vol) if ann_vol > 0 else 0.0
    curve   = np.cumprod(1 + rets)
    mdd     = float((curve / np.maximum.accumulate(curve) - 1).min())
    calmar  = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0
    return {"name": name, "cum": cum, "cum_ret": cum,
            "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "mdd": mdd, "calmar": calmar,
            "_port_ret": rets}


def mvo_metrics(weights: np.ndarray, name: str) -> dict:
    rets = []
    prev = np.zeros(weights.shape[1])
    for w, row in zip(weights, rows_2022):
        asset_rets = np.array([
            get_period_return(prices_by_asset[a], row["input_end_date"], row["target_date"])
            for a in ASSETS
        ])
        turnover = float(np.sum(np.abs(w - prev)))
        rets.append(float(w @ asset_rets - turnover * COST))
        prev = w
    rets = np.array(rets)
    cum     = float(np.prod(1 + rets) - 1)
    ann_ret = float((1 + cum) ** (PPY / len(rets)) - 1)
    ann_vol = float(rets.std() * np.sqrt(PPY))
    sharpe  = float((ann_ret - RF) / ann_vol) if ann_vol > 0 else 0.0
    curve   = np.cumprod(1 + rets)
    mdd     = float((curve / np.maximum.accumulate(curve) - 1).min())
    calmar  = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0
    return {"name": name, "cum": cum, "cum_ret": cum,
            "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "mdd": mdd, "calmar": calmar,
            "_port_ret": rets}


def get_sma(end_date: str, window: int) -> float:
    idx = spy_dates.index(end_date) if end_date in spy_dates else -1
    if idx < window:
        return spy_prices[end_date]
    return float(np.mean([spy_prices[d] for d in spy_dates[idx - window + 1:idx + 1]]))


ma_weights = np.array([
    1.0 if get_sma(r["input_end_date"], 20) > get_sma(r["input_end_date"], 60) else 0.0
    for r in rows_2022
])

# 단순 SPY/Cash: classifier signal only, no MVO.
# Bull 확률 + Neutral 절반 → SPY 비중, 나머지 현금
w_spy_cash = probs_2022[:, 2] + 0.5 * probs_2022[:, 1]  # (n,)

results = {
    "Buy & Hold":                spy_metrics(np.ones(n),       spy_rets, "Buy & Hold"),
    "60/40":                     spy_metrics(np.full(n, 0.6),  spy_rets, "60/40"),
    "MA Crossover":              spy_metrics(ma_weights,       spy_rets, "MA Crossover"),
    "EW (1/N)":                  spy_metrics(np.ones(n),       ew_rets,  "EW (1/N)"),
    "DL Regime SPY/Cash":        spy_metrics(w_spy_cash,       spy_rets, "DL Regime SPY/Cash"),
    "Regime-MVO (ours)":         mvo_metrics(regime_mvo_weights, "Regime-MVO (ours)"),
    "Regime Momentum Tilt":      rmt_metrics(rmt_weights, "Regime Momentum Tilt"),
}


# ── 7. 결과 출력 ──────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"2022 하락장 백테스트: {rows_2022[0]['input_end_date']} ~ {rows_2022[-1]['target_date']}")
print(f"{'='*72}")
print(f"{'전략':<26} {'누적수익':>8} {'연수익':>7} {'변동성':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7}")
print("─" * 72)
for name, m in results.items():
    marker = " ◀" if "Regime" in name else ""
    print(f"{name:<26} {m['cum']:>7.1%}  {m['ann_ret']:>6.1%}  "
          f"{m['ann_vol']:>6.1%}  {m['sharpe']:>6.2f}  {m['mdd']:>7.1%}  "
          f"{m['calmar']:>6.2f}{marker}")


# ── 8. JSON 저장 (_port_ret 제외) ────────────────────────────────
save = {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
             for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in results.items()}
Path("outputs/results").mkdir(parents=True, exist_ok=True)
with open("outputs/results/backtest_2022_results.json", "w") as f:
    json.dump(save, f, indent=2, ensure_ascii=False)
print("\n결과 저장: outputs/results/backtest_2022_results.json")


# ── 9. Fig 5: 누적 수익률 + MDD 막대 ─────────────────────────────
style = {
    "Buy & Hold":                ("#2C3E50", "--", 1.5),
    "60/40":                     ("#95A5A6", "--", 1.2),
    "MA Crossover":              ("#8E44AD", ":",  1.2),
    "EW (1/N)":                  ("#27AE60", "--", 1.5),
    "DL Regime SPY/Cash":        ("#2980B9", "-",  2.8),
    "Regime-MVO (ours)":         ("#E74C3C", "-",  2.4),
    "Regime Momentum Tilt":      ("#E67E22", "-",  1.8),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"2022 Bear Market Backtest\n(SPY -18.6%,  {rows_2022[0]['input_end_date']} ~ {rows_2022[-1]['target_date']})",
    fontsize=13, fontweight="bold"
)

# 누적 수익률 곡선
dates_axis = [r["target_date"] for r in rows_2022]
for name, m in results.items():
    color, ls, lw = style[name]
    curve = np.concatenate([[1.0], np.cumprod(1 + m["_port_ret"])])
    lbl = f"{name} ◀" if name == "Regime-MVO (ours)" else name
    ax1.plot(range(len(curve)), (curve - 1) * 100,
             color=color, linestyle=ls, linewidth=lw, label=lbl)

ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_ylabel("누적 수익률 (%)")
ax1.set_title("누적 수익률 비교")
tick_idx = list(range(0, len(dates_axis) + 1, 8))
tick_lbl = ["Start"] + [dates_axis[min(i - 1, len(dates_axis) - 1)][:7] for i in tick_idx[1:]]
ax1.set_xticks(tick_idx[:len(tick_lbl)])
ax1.set_xticklabels(tick_lbl, rotation=30, fontsize=8)
ax1.legend(fontsize=9)

# MDD 막대
names_order = list(results.keys())
short_names = [
    "Buy &\nHold", "60/40", "MA\nCrossover", "EW\n1/N",
    "DL Regime\nSPY/Cash", "Regime-\nMVO ◀", "Regime\nTilt",
]
mdds = [abs(results[n]["mdd"]) * 100 for n in names_order]
bar_colors = ["#E74C3C" if "Regime-MVO" in n else "#2980B9" if "DL Regime" in n else "#E67E22" if "Regime" in n
              else "#27AE60" if "EW" in n else "#95A5A6"
              for n in names_order]

bars = ax2.bar(short_names, mdds, color=bar_colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, mdds):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
             f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax2.set_ylabel("Max Drawdown (%)")
ax2.set_title("Max Drawdown 비교\n(낮을수록 좋음)")
# 가장 낮은 MDD 강조
best_idx = mdds.index(min(mdds))
bars[best_idx].set_edgecolor("gold")
bars[best_idx].set_linewidth(2.5)

plt.tight_layout()
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/fig5_2022_bear_backtest.png", bbox_inches="tight")
plt.close()
print("그래프 저장: outputs/figures/fig5_2022_bear_backtest.png")

"""
Regime-Conditioned MVO Backtest

훈련셋에서 국면별(Bear/Neutral/Bull) 최적 MVO 비중 계산 →
테스트에서 모델 확률로 소프트 배분:
    w = p_bear * w_bear + p_neutral * w_neutral + p_bull * w_bull

출력: outputs/results/backtest_mvo_results.json
      outputs/figures/fig7_mvo_comparison.png
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
from regime_portfolio_policy import ASSETS, get_period_return

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

COST = 0.001
RF   = 0.05
PPY  = 252 / 5
RF_P = RF / PPY   # 기간별 무위험수익률


# ── 1. 데이터 로드 ────────────────────────────────────────────────
data       = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
y_train    = data["y_train"]   # (488,)  0=Bear 1=Neutral 2=Bull
X_test_np  = data["X_test"].astype(np.float32)

rows_all   = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
train_rows = [r for r in rows_all if r["split"] == "train"]
test_rows  = [r for r in rows_all if r["split"] == "test"]


def load_prices(path: str) -> dict:
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row["Date"]] = float(row["Adj Close"])
    return d

prices = {a: load_prices(f"data/raw/{a.lower()}_daily.csv") for a in ASSETS}


# ── 2. 수익률 행렬 계산 ───────────────────────────────────────────
def returns_matrix(rows, prices_dict):
    """(n, 4) — SPY/QQQ/GLD/TLT 5일 수익률"""
    R = np.zeros((len(rows), len(ASSETS)))
    for i, row in enumerate(rows):
        for j, asset in enumerate(ASSETS):
            R[i, j] = get_period_return(prices_dict[asset],
                                        row["input_end_date"], row["target_date"])
    return R

train_R = returns_matrix(train_rows, prices)   # (488, 4)
test_R  = returns_matrix(test_rows,  prices)   # (105, 4)


# ── 3. 국면별 MVO: Sharpe 최대화 ─────────────────────────────────
def max_sharpe_weights(R: np.ndarray, rf: float = RF_P) -> np.ndarray:
    """Long-only, 합계=1 제약 하에서 Sharpe 최대화"""
    n = R.shape[1]
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        port = R @ w
        mu   = port.mean() - rf
        sig  = port.std(ddof=0)
        return -(mu / sig) if sig > 1e-8 else 0.0

    result = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return result.x if result.success else w0


REGIME_NAMES = {0: "Bear", 1: "Neutral", 2: "Bull"}
mvo_w = {}

print("\n[국면별 MVO 최적 비중]")
print(f"{'국면':<10} {'샘플':>6}  {'SPY':>7} {'QQQ':>7} {'GLD':>7} {'TLT':>7}")
print("─" * 50)
for rid in [0, 1, 2]:
    mask = y_train == rid
    R_r  = train_R[mask]
    w    = max_sharpe_weights(R_r)
    mvo_w[rid] = w
    print(f"{REGIME_NAMES[rid]:<10} {mask.sum():>6}  "
          + "  ".join(f"{v:>6.1%}" for v in w))


# ── 4. 모델 예측 ──────────────────────────────────────────────────
model = RegimeClassifier(input_size=40, conv_channels=16, lstm_hidden=32).to(device)
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
model.eval()
with torch.no_grad():
    probs = model.predict_proba(
        torch.tensor(X_test_np).to(device)
    ).cpu().numpy()   # (105, 3)


# ── 5. 소프트 MVO 비중: p_bear*w_bear + p_neutral*w_neutral + p_bull*w_bull ──
soft_weights = (
    probs[:, 0:1] * mvo_w[0]
    + probs[:, 1:2] * mvo_w[1]
    + probs[:, 2:3] * mvo_w[2]
)   # (105, 4)


# ── 6. 성과 계산 ──────────────────────────────────────────────────
def portfolio_metrics(weights: np.ndarray, R: np.ndarray, name: str) -> dict:
    """weights: (n,4) 또는 (n,) SPY 단독"""
    rets = []
    prev = np.zeros(weights.shape[1] if weights.ndim == 2 else 1)
    for t, (w, r) in enumerate(zip(weights, R)):
        turnover = float(np.sum(np.abs(w - prev)))
        rets.append(float(w @ r - turnover * COST))
        prev = w
    rets = np.array(rets)
    cum     = float(np.prod(1 + rets) - 1)
    ann_ret = float((1 + cum) ** (PPY / len(rets)) - 1)
    ann_vol = float(rets.std() * np.sqrt(PPY))
    sharpe  = float((ann_ret - RF) / ann_vol) if ann_vol > 0 else 0.0
    curve   = np.cumprod(1 + rets)
    mdd     = float((curve / np.maximum.accumulate(curve) - 1).min())
    calmar  = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0
    return {"name": name, "cum_ret": cum, "ann_ret": ann_ret,
            "ann_vol": ann_vol, "sharpe": sharpe, "mdd": mdd,
            "calmar": calmar, "_rets": rets}


# EW 1/N (SPY/QQQ/GLD/TLT 균등)
ew_w = np.tile(np.ones(4) / 4, (len(test_rows), 1))

# Buy & Hold (SPY 100%)
bnh_w = np.zeros((len(test_rows), 4))
bnh_w[:, 0] = 1.0

# 60/40 SPY/Cash → SPY 60%
s6040_w = np.zeros((len(test_rows), 4))
s6040_w[:, 0] = 0.60

# DL Regime SPY/Cash: classifier signal only, no MVO
lstm_w_spy = probs[:, 2] + 0.5 * probs[:, 1]
lstm_w = np.zeros((len(test_rows), 4))
lstm_w[:, 0] = lstm_w_spy

# Regime-Agnostic MVO: 국면 무시, 전체 훈련셋 단일 MVO (국면 conditioning 효과 비교용)
agnostic_w_single = max_sharpe_weights(train_R)
agnostic_w = np.tile(agnostic_w_single, (len(test_rows), 1))

# Oracle: HMM pseudo-label(y_test)로 비중 결정 (분류기 상한선)
data_full   = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
y_test_true = data_full["y_test"]
oracle_w    = np.array([mvo_w[int(y)] for y in y_test_true])

results = {
    "Buy & Hold":             portfolio_metrics(bnh_w,        test_R, "Buy & Hold"),
    "EW 1/N":                 portfolio_metrics(ew_w,         test_R, "EW 1/N"),
    "60/40":                  portfolio_metrics(s6040_w,      test_R, "60/40"),
    "Regime-Agnostic MVO":    portfolio_metrics(agnostic_w,   test_R, "Regime-Agnostic MVO"),
    "DL Regime SPY/Cash":     portfolio_metrics(lstm_w,       test_R, "DL Regime SPY/Cash"),
    "Regime-MVO (ours)":      portfolio_metrics(soft_weights, test_R, "Regime-MVO (ours)"),
    "Oracle (HMM labels)":    portfolio_metrics(oracle_w,     test_R, "Oracle (HMM labels)"),
}


# ── 7. 결과 출력 ──────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"백테스트: {test_rows[0]['input_end_date']} ~ {test_rows[-1]['target_date']}")
print(f"{'='*72}")
print(f"{'전략':<26} {'누적':>7} {'연수익':>7} {'변동성':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7}")
print("─" * 72)
for name, m in results.items():
    marker = " <- ours" if "ours" in name else ""
    print(f"{name:<26} {m['cum_ret']:>6.1%}  {m['ann_ret']:>6.1%}  "
          f"{m['ann_vol']:>6.1%}  {m['sharpe']:>6.2f}  {m['mdd']:>7.1%}  "
          f"{m['calmar']:>6.2f}{marker}")

# 핵심 비교 출력
print()
r_agnostic = results["Regime-Agnostic MVO"]
r_ours     = results["Regime-MVO (ours)"]
r_oracle   = results["Oracle (HMM labels)"]
print(f"[국면 conditioning 효과] MDD {r_agnostic['mdd']:.1%} → {r_ours['mdd']:.1%}"
      f"  ({abs(r_ours['mdd']-r_agnostic['mdd'])*100:.1f}pp 개선)")
print(f"[분류기 개선 여지]       MDD {r_ours['mdd']:.1%} → {r_oracle['mdd']:.1%}"
      f"  (Oracle까지 {abs(r_oracle['mdd']-r_ours['mdd'])*100:.1f}pp 남음)")


# ── 8. JSON 저장 ──────────────────────────────────────────────────
Path("outputs/results").mkdir(parents=True, exist_ok=True)
save = {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
             for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in results.items()}
with open("outputs/results/backtest_mvo_results.json", "w") as f:
    json.dump(save, f, indent=2, ensure_ascii=False)


# ── 9. Fig 7: 누적 수익률 곡선 비교 ─────────────────────────────
STYLE = {
    "Buy & Hold":           ("#7F8C8D", "--", 1.5),
    "EW 1/N":               ("#27AE60", "--", 1.5),
    "60/40":                ("#BDC3C7", "--", 1.2),
    "Regime-Agnostic MVO":  ("#E67E22", "--", 1.8),
    "DL Regime SPY/Cash":   ("#2980B9", "-",  1.8),
    "Regime-MVO (ours)":    ("#E74C3C", "-",  2.8),
    "Oracle (HMM labels)":  ("#8E44AD", ":",  2.0),
}

fig, ax = plt.subplots(figsize=(13, 5.5))
dates_axis = [r["target_date"] for r in test_rows]

for name, m in results.items():
    color, ls, lw = STYLE[name]
    curve = np.concatenate([[1.0], np.cumprod(1 + m["_rets"])])
    if "ours" in name:
        lbl = "Regime-MVO (ours)"
    elif "Oracle" in name:
        lbl = "Oracle (HMM labels, upper bound)"
    else:
        lbl = name
    ax.plot(range(len(curve)), (curve - 1) * 100,
            color=color, linestyle=ls, linewidth=lw, label=lbl)

# MDD 비교 주석
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Rebalancing Period (5-day intervals)")
ax.set_ylabel("Cumulative Return (%)")
ax.set_title(
    "Regime-Conditioned MVO vs Baselines  (2024.04 ~ 2026.05)\n"
    "국면 conditioning: MDD -20.8% → -7.2%  |  Oracle 상한: -6.2%",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")

tick_idx = list(range(0, len(dates_axis) + 1, 20))
tick_lbl = ["Start"] + [dates_axis[i - 1][:7] for i in tick_idx[1:] if i <= len(dates_axis)]
ax.set_xticks(tick_idx[:len(tick_lbl)])
ax.set_xticklabels(tick_lbl, rotation=30, fontsize=8)

plt.tight_layout()
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/fig_mvo_full_comparison.png", bbox_inches="tight", dpi=130)
plt.close()

print("\n저장: outputs/results/backtest_mvo_results.json")
print("저장: outputs/figures/fig_mvo_full_comparison.png")

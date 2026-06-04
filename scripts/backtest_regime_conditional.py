"""
Regime-conditional strategy performance.

Bear / Neutral / Bull periods are split by HMM pseudo-labels in y_test.
This is a diagnostic figure, not proof that the HMM states are true market labels.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier
from regime_portfolio_policy import ASSETS, get_period_return

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COST = 0.001
RF   = 0.05
PPY  = 252 / 5
RF_P = RF / PPY

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# ── 1. 데이터 로드 ────────────────────────────────────────────────
data      = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
y_train   = data["y_train"]
y_test    = data["y_test"]          # HMM pseudo-labels (0=Bear, 1=Neutral, 2=Bull)
X_test_np = data["X_test"].astype(np.float32)

rows_all   = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
train_rows = [r for r in rows_all if r["split"] == "train"]
test_rows  = [r for r in rows_all if r["split"] == "test"]

def load_prices(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row["Date"]] = float(row["Adj Close"])
    return d

prices = {a: load_prices(f"data/raw/{a.lower()}_daily.csv") for a in ASSETS}

def returns_matrix(rows, prices_dict):
    R = np.zeros((len(rows), len(ASSETS)))
    for i, row in enumerate(rows):
        for j, asset in enumerate(ASSETS):
            R[i, j] = get_period_return(prices_dict[asset],
                                        row["input_end_date"], row["target_date"])
    return R

train_R = returns_matrix(train_rows, prices)
test_R  = returns_matrix(test_rows,  prices)


# ── 2. 국면별 MVO 비중 (훈련셋 Sharpe 최대화) ─────────────────────
def max_sharpe_weights(R, rf=RF_P):
    n = R.shape[1]
    w0 = np.ones(n) / n
    def neg_sharpe(w):
        port = R @ w
        mu   = port.mean() - rf
        sig  = port.std(ddof=0)
        return -(mu / sig) if sig > 1e-8 else 0.0
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return res.x if res.success else w0

mvo_w = {rid: max_sharpe_weights(train_R[y_train == rid]) for rid in [0, 1, 2]}


# ── 3. 모델 예측 ──────────────────────────────────────────────────
model = RegimeClassifier(input_size=40, conv_channels=16, lstm_hidden=32).to(device)
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
model.eval()
with torch.no_grad():
    probs = model.predict_proba(
        torch.tensor(X_test_np).to(device)
    ).cpu().numpy()   # (105, 3)


# ── 4. 전략별 기간 수익률 ─────────────────────────────────────────
n = len(test_rows)

bnh_w_mat  = np.zeros((n, 4)); bnh_w_mat[:, 0]  = 1.0
ew_w_mat   = np.tile(np.ones(4) / 4, (n, 1))
s6040_mat  = np.zeros((n, 4)); s6040_mat[:, 0]  = 0.6
lstm_mat   = np.zeros((n, 4))
lstm_mat[:, 0] = probs[:, 2] + 0.5 * probs[:, 1]
mvo_mat    = (probs[:, 0:1] * mvo_w[0]
            + probs[:, 1:2] * mvo_w[1]
            + probs[:, 2:3] * mvo_w[2])

def period_rets(weights, R, cost=COST):
    out, prev = [], np.zeros(4)
    for w, r in zip(weights, R):
        out.append(float(w @ r - np.sum(np.abs(w - prev)) * cost))
        prev = w
    return np.array(out)

rets = {
    "Buy & Hold":             period_rets(bnh_w_mat,  test_R),
    "EW 1/N":                 period_rets(ew_w_mat,   test_R),
    "60/40":                  period_rets(s6040_mat,  test_R),
    "DL Regime SPY/Cash":     period_rets(lstm_mat,   test_R),
    "Regime-MVO (ours)":      period_rets(mvo_mat,    test_R),
}


# ── 5. 국면별 연환산 수익률 계산 ──────────────────────────────────
def ann_ret(r_sub):
    if len(r_sub) == 0:
        return 0.0
    cum = float(np.prod(1 + r_sub) - 1)
    return ((1 + cum) ** (PPY / len(r_sub)) - 1) * 100

REGIME_NAMES = ["Bear", "Neutral", "Bull"]
counts = {i: int((y_test == i).sum()) for i in range(3)}

print("\n[국면별 샘플 수]")
for i, name in enumerate(REGIME_NAMES):
    print(f"  {name}: {counts[i]} periods")

print("\n[국면별 연환산 수익률 (%)]")
print(f"{'전략':<28} {'Bear':>8} {'Neutral':>9} {'Bull':>8}")
print("─" * 57)

regime_data = {}
for sname, srets in rets.items():
    regime_data[sname] = [ann_ret(srets[y_test == rid]) for rid in range(3)]
    print(f"{sname:<28} " + "  ".join(f"{v:>8.1f}" for v in regime_data[sname]))


# ── 6. MDD per regime ────────────────────────────────────────────
def regime_mdd(r_arr, mask):
    r = r_arr[mask]
    if len(r) == 0:
        return 0.0
    curve = np.cumprod(1 + r)
    return float(abs((curve / np.maximum.accumulate(curve) - 1).min())) * 100

mdd_data = {
    sname: [regime_mdd(srets, y_test == rid) for rid in range(3)]
    for sname, srets in rets.items()
}

print("\n[국면별 MDD (%)]")
print(f"{'전략':<28} {'Bear':>8} {'Neutral':>9} {'Bull':>8}")
print("─" * 57)
for sname, vals in mdd_data.items():
    print(f"{sname:<28} " + "  ".join(f"{v:>8.1f}" for v in vals))


# ── 7. 플롯: 2행 × 1열 (상=수익률, 하=MDD) ───────────────────────
STRAT_STYLE = {
    "Buy & Hold":             "#7F8C8D",
    "EW 1/N":                 "#27AE60",
    "60/40":                  "#BDC3C7",
    "DL Regime SPY/Cash":     "#2980B9",
    "Regime-MVO (ours)":      "#E74C3C",
}
SHORT = {
    "Buy & Hold":             "Buy &\nHold",
    "EW 1/N":                 "EW 1/N",
    "60/40":                  "60/40",
    "DL Regime SPY/Cash":     "DL Regime\nSPY/Cash",
    "Regime-MVO (ours)":      "Regime-\nMVO\n(ours)",
}

x     = np.arange(3)
n_s   = len(STRAT_STYLE)
w_bar = 0.15
offs  = np.linspace(-(n_s - 1) / 2, (n_s - 1) / 2, n_s) * w_bar

xlabels = [f"{rname}\n({counts[i]} periods)"
           for i, rname in enumerate(REGIME_NAMES)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))
fig.suptitle(
    "Regime-Conditional Strategy Performance\n"
    "HMM pseudo-label groups, test period: 2024.04 ~ 2026.05",
    fontsize=14, fontweight="bold", y=0.98,
)

RET_CAP = 52   # Neutral B&H 75.2% → 잘라서 표시

# ── 상단: 연환산 수익률 ───────────────────────────────────────────
for i, (sname, color) in enumerate(STRAT_STYLE.items()):
    vals = regime_data[sname]
    plot_vals = [min(v, RET_CAP) for v in vals]   # cap for display
    bars = ax1.bar(x + offs[i], plot_vals, w_bar,
                   label=SHORT[sname].replace("\n", " "),
                   color=color, alpha=0.85, zorder=3)
    for bar, raw_v, plot_v in zip(bars, vals, plot_vals):
        clipped = raw_v > RET_CAP
        ypos = bar.get_height() + 0.5 if raw_v >= 0 else bar.get_height() - 1.5
        va   = "bottom" if raw_v >= 0 else "top"
        label = f"{raw_v:.1f}%↑" if clipped else f"{raw_v:.1f}%"
        ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                 label, ha="center", va=va,
                 fontsize=7.5, fontweight="bold", color=color)

# Bear period: Regime-MVO vs B&H diagnostic comparison
bear_mvo = regime_data["Regime-MVO (ours)"][0]
bear_bnh = regime_data["Buy & Hold"][0]
diff_bear = bear_mvo - bear_bnh
ax1.annotate(
    f"+{diff_bear:.1f}pp\nvs B&H",
    xy=(x[0] + offs[-1], bear_mvo),
    xytext=(x[0] + offs[-1] + 0.25, bear_mvo + 4),
    fontsize=9, color="#E74C3C", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor="#E74C3C"),
    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.3),
)

ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_ylim(-15, RET_CAP + 5)
ax1.set_xticks(x); ax1.set_xticklabels(xlabels, fontsize=12, fontweight="bold")
ax1.set_ylabel("Annualized Return (%)", fontsize=11)
ax1.set_title("(a) Annualized Return by Market Regime", fontsize=11, fontweight="bold", pad=8)
ax1.legend(fontsize=9, loc="upper right", ncol=3, framealpha=0.9)

# ── 하단: MDD ────────────────────────────────────────────────────
for i, (sname, color) in enumerate(STRAT_STYLE.items()):
    vals = mdd_data[sname]
    bars = ax2.bar(x + offs[i], vals, w_bar,
                   label=SHORT[sname].replace("\n", " "),
                   color=color, alpha=0.85, zorder=3)
    # 국면별 최솟값(최저 MDD)에 금색 테두리
    for rid in range(3):
        best_val = min(mdd_data[s][rid] for s in STRAT_STYLE)
        if abs(vals[rid] - best_val) < 0.05:
            bars[rid].set_edgecolor("gold")
            bars[rid].set_linewidth(2.5)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold", color=color)

# Bear period MDD diagnostic comparison
bear_mdd_mvo = mdd_data["Regime-MVO (ours)"][0]
bear_mdd_bnh = mdd_data["Buy & Hold"][0]
mdd_diff = bear_mdd_bnh - bear_mdd_mvo   # positive = MVO has lower MDD
ax2.annotate(
    f"-{mdd_diff:.1f}pp MDD\nvs B&H",
    xy=(x[0] + offs[-1], bear_mdd_mvo),
    xytext=(x[0] + offs[-1] + 0.25, bear_mdd_mvo + 3),
    fontsize=9, color="#E74C3C", fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor="#E74C3C"),
    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.3),
)

ax2.set_xticks(x); ax2.set_xticklabels(xlabels, fontsize=12, fontweight="bold")
ax2.set_ylabel("Max Drawdown (%)", fontsize=11)
ax2.set_title("(b) Max Drawdown by Market Regime  (lower = better)", fontsize=11, fontweight="bold", pad=8)
ax2.legend(fontsize=9, loc="upper right", ncol=3, framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
Path("outputs/figures/final").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/final/fig06_regime_conditional.png", bbox_inches="tight")
plt.close()
print("\n저장: outputs/figures/final/fig06_regime_conditional.png")

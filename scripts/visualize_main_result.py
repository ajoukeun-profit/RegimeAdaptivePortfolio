"""
fig06 — 핵심 결과: 국면 인식이 포트폴리오를 지킨다
fig07 — Bear 구간 심층 분석
"""
import csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier
from regime_portfolio_policy import ASSETS, get_period_return
from scipy.optimize import minimize

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})

COST=0.001; RF=0.05; PPY=252/5; RF_P=RF/PPY
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ── 데이터 로드 ───────────────────────────────────────────────────
data      = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
y_train   = data["y_train"]
y_test    = data["y_test"]
X_test_np = data["X_test"].astype(np.float32)

rows_all   = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
train_rows = [r for r in rows_all if r["split"] == "train"]
test_rows  = [r for r in rows_all if r["split"] == "test"]

def load_prices(p):
    d = {}
    with open(p) as f:
        for row in csv.DictReader(f): d[row["Date"]] = float(row["Adj Close"])
    return d
prices = {a: load_prices(f"data/raw/{a.lower()}_daily.csv") for a in ASSETS}

def returns_matrix(rows):
    R = np.zeros((len(rows), len(ASSETS)))
    for i, row in enumerate(rows):
        for j, asset in enumerate(ASSETS):
            R[i, j] = get_period_return(prices[asset], row["input_end_date"], row["target_date"])
    return R
train_R = returns_matrix(train_rows)
test_R  = returns_matrix(test_rows)

def max_sharpe(R, rf=RF_P):
    n=R.shape[1]; w0=np.ones(n)/n
    def ns(w): p=R@w; m=p.mean()-rf; s=p.std(ddof=0); return -(m/s) if s>1e-8 else 0.
    res=minimize(ns,w0,method="SLSQP",bounds=[(0,1)]*n,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}])
    return res.x if res.success else w0

mvo_w = {rid: max_sharpe(train_R[y_train==rid]) for rid in [0,1,2]}

model = RegimeClassifier(input_size=40,conv_channels=16,lstm_hidden=32,num_classes=3).to(device)
model.load_state_dict(torch.load("outputs/models/best_model.pt",map_location=device))
model.eval()
with torch.no_grad():
    probs = model.predict_proba(torch.tensor(X_test_np).to(device)).cpu().numpy()

n = len(test_rows)
bnh_w       = np.zeros((n,4)); bnh_w[:,0] = 1.0
ew_w        = np.tile(np.ones(4)/4,(n,1))
agnostic_w  = np.tile(max_sharpe(train_R),(n,1))
soft_w      = probs[:,0:1]*mvo_w[0] + probs[:,1:2]*mvo_w[1] + probs[:,2:3]*mvo_w[2]
oracle_w    = np.array([mvo_w[int(y)] for y in y_test])

def port_rets(weights, R):
    rets, prev = [], np.zeros(4)
    for w, r in zip(weights, R):
        rets.append(float(w@r - np.sum(np.abs(w-prev))*COST)); prev=w
    return np.array(rets)

def metrics(rets):
    cum = float(np.prod(1+rets)-1)
    ann = float((1+cum)**(PPY/len(rets))-1)
    vol = float(rets.std()*np.sqrt(PPY))
    sharpe = float((ann-RF)/vol) if vol>0 else 0.
    curve = np.cumprod(1+rets)
    mdd = float((curve/np.maximum.accumulate(curve)-1).min())
    calmar = float(ann/abs(mdd)) if mdd<0 else 0.
    return {"cum":cum,"sharpe":sharpe,"mdd":mdd,"calmar":calmar,"rets":rets}

STRATS = {
    "Buy &\nHold":          metrics(port_rets(bnh_w,    test_R)),
    "EW 1/N":               metrics(port_rets(ew_w,     test_R)),
    "Agnostic\nMVO":        metrics(port_rets(agnostic_w,test_R)),
    "Regime-MVO\n(ours) ★": metrics(port_rets(soft_w,   test_R)),
    "Oracle\n(상한선)":     metrics(port_rets(oracle_w, test_R)),
}

COLORS = {
    "Buy &\nHold":          "#7F8C8D",
    "EW 1/N":               "#27AE60",
    "Agnostic\nMVO":        "#E67E22",
    "Regime-MVO\n(ours) ★": "#E74C3C",
    "Oracle\n(상한선)":     "#8E44AD",
}

names  = list(STRATS.keys())
mdds   = [abs(STRATS[n]["mdd"])*100  for n in names]
calmars= [STRATS[n]["calmar"]        for n in names]
colors = [COLORS[n]                  for n in names]

# ══════════════════════════════════════════════════════════════════
# Fig 06 — 핵심 결과
# ══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "국면 인식이 포트폴리오를 지킨다\n"
    "Regime Conditioning Reduces Drawdown  (Test: 2024.04 ~ 2026.05)",
    fontsize=13, fontweight="bold", y=1.01
)

x = np.arange(len(names))

# ── 왼쪽: MDD ────────────────────────────────────────────────────
bars1 = ax1.bar(x, mdds, color=colors, alpha=0.88, width=0.6, zorder=3)
# 최솟값 금색 테두리
best_idx = mdds.index(min(mdds))
bars1[best_idx].set_edgecolor("gold"); bars1[best_idx].set_linewidth(3)

for bar, val, c in zip(bars1, mdds, colors):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=c)

# 핵심 어노테이션
ax1.annotate("",
    xy=(x[3], mdds[3]+0.3), xytext=(x[2], mdds[2]+0.3),
    arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=2))
ax1.text((x[2]+x[3])/2, max(mdds[2],mdds[3])+2.0,
         f"{mdds[2]-mdds[3]:.0f}pp 개선\n(국면 conditioning 효과)",
         ha="center", fontsize=9, fontweight="bold", color="#2C3E50",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDFEFE", edgecolor="#2C3E50", alpha=0.85))

ax1.annotate("",
    xy=(x[4], mdds[4]+0.3), xytext=(x[3], mdds[3]+0.3),
    arrowprops=dict(arrowstyle="<->", color="#8E44AD", lw=1.5, linestyle="dashed"))
ax1.text((x[3]+x[4])/2, max(mdds[3],mdds[4])+2.0,
         f"{mdds[3]-mdds[4]:.1f}pp\nOracle까지",
         ha="center", fontsize=8.5, color="#8E44AD",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5EEF8", edgecolor="#8E44AD", alpha=0.85))

ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9.5)
ax1.set_ylabel("Max Drawdown (%)", fontsize=11)
ax1.set_title("(a) Max Drawdown  (낮을수록 좋음)", fontsize=11, fontweight="bold")
ax1.set_ylim(0, max(mdds)*1.45)

# ── 오른쪽: Calmar ───────────────────────────────────────────────
bars2 = ax2.bar(x, calmars, color=colors, alpha=0.88, width=0.6, zorder=3)
best_c = calmars.index(max(calmars))
bars2[best_c].set_edgecolor("gold"); bars2[best_c].set_linewidth(3)

for bar, val, c in zip(bars2, calmars, colors):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=c)

ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9.5)
ax2.set_ylabel("Calmar Ratio  (수익 ÷ MDD)", fontsize=11)
ax2.set_title("(b) Calmar Ratio  (높을수록 좋음)", fontsize=11, fontweight="bold")

# 범례
leg = [
    mpatches.Patch(color="#E74C3C", label="Regime-MVO (ours)"),
    mpatches.Patch(color="#E67E22", label="Agnostic MVO (국면 무시)"),
    mpatches.Patch(color="#8E44AD", label="Oracle (완벽한 분류기 상한)"),
    mpatches.Patch(color="#27AE60", label="EW 1/N"),
    mpatches.Patch(color="#7F8C8D", label="Buy & Hold"),
]
fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

plt.tight_layout()
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/fig06_main_result.png", bbox_inches="tight")
plt.close()
print("저장: fig06_main_result.png")


# ══════════════════════════════════════════════════════════════════
# Fig 07 — Bear 구간 집중 분석
# ══════════════════════════════════════════════════════════════════
def regime_ann_ret(rets, mask):
    r = rets[mask]
    if len(r) == 0: return 0.
    cum = float(np.prod(1+r)-1)
    return ((1+cum)**(PPY/len(r))-1)*100

def regime_mdd(rets, mask):
    r = rets[mask]
    if len(r) == 0: return 0.
    curve = np.cumprod(1+r)
    return abs(float((curve/np.maximum.accumulate(curve)-1).min()))*100

bear_mask = y_test == 0

STRAT_FULL = {
    "Buy &\nHold":          port_rets(bnh_w,     test_R),
    "EW 1/N":               port_rets(ew_w,      test_R),
    "Agnostic\nMVO":        port_rets(agnostic_w,test_R),
    "Regime-MVO\n(ours) ★": port_rets(soft_w,    test_R),
    "Oracle\n(상한선)":     port_rets(oracle_w,  test_R),
}

bear_rets = [regime_ann_ret(v, bear_mask) for v in STRAT_FULL.values()]
bear_mdds = [regime_mdd(v, bear_mask)    for v in STRAT_FULL.values()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    f"Bear 구간 심층 분석  ({(bear_mask).sum()} periods, 2024.04 ~ 2026.05)\n"
    "하락장에서 각 전략이 얼마나 버텼는가",
    fontsize=13, fontweight="bold", y=1.01
)

# ── 왼쪽: Bear 구간 연환산 수익률 ────────────────────────────────
bars1 = ax1.bar(x, bear_rets, color=colors, alpha=0.88, width=0.6, zorder=3)
ax1.axhline(0, color="black", linewidth=0.8)
for bar, val, c in zip(bars1, bear_rets, colors):
    ypos = bar.get_height()+(0.3 if val>=0 else -1.2)
    va = "bottom" if val >= 0 else "top"
    ax1.text(bar.get_x()+bar.get_width()/2, ypos, f"{val:.1f}%",
             ha="center", va=va, fontsize=10, fontweight="bold", color=c)
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9.5)
ax1.set_ylabel("Annualized Return (%)", fontsize=11)
ax1.set_title("(a) Bear 구간 연환산 수익률", fontsize=11, fontweight="bold")

# ── 오른쪽: Bear 구간 MDD ────────────────────────────────────────
bars2 = ax2.bar(x, bear_mdds, color=colors, alpha=0.88, width=0.6, zorder=3)
best_b = bear_mdds.index(min(bear_mdds))
bars2[best_b].set_edgecolor("gold"); bars2[best_b].set_linewidth(3)
for bar, val, c in zip(bars2, bear_mdds, colors):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=c)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9.5)
ax2.set_ylabel("Max Drawdown (%)", fontsize=11)
ax2.set_title("(b) Bear 구간 MDD  (낮을수록 좋음)", fontsize=11, fontweight="bold")

fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

plt.tight_layout()
plt.savefig("outputs/figures/fig07_bear_analysis.png", bbox_inches="tight")
plt.close()
print("저장: fig07_bear_analysis.png")

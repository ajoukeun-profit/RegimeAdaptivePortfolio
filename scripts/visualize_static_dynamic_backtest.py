"""
Presentation figures focused on the actual research question:
static benchmark portfolios vs dynamic Regime-MVO.

Outputs:
  outputs/figures/final/fig03_static_dynamic_backtest.png
  outputs/figures/final/fig03_main_result.png
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from regime_portfolio_policy import ASSETS, get_period_return
from train import RegimeClassifier


plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})

COST = 0.001
RF = 0.05
PPY = 252 / 5
RF_P = RF / PPY

OUT_DIR = Path("outputs/figures/final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Buy & Hold": "#7F8C8D",
    "EW 1/N": "#27AE60",
    "60/40": "#BDC3C7",
    "Regime-Agnostic MVO": "#E67E22",
    "DL Regime SPY/Cash": "#2980B9",
    "Regime-MVO": "#E74C3C",
    "Oracle (HMM labels)": "#8E44AD",
}


def load_prices(path: str) -> dict[str, float]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["Date"]] = float(row["Adj Close"])
    return out


def returns_matrix(rows, prices_dict):
    r = np.zeros((len(rows), len(ASSETS)))
    for i, row in enumerate(rows):
        for j, asset in enumerate(ASSETS):
            r[i, j] = get_period_return(
                prices_dict[asset],
                row["input_end_date"],
                row["target_date"],
            )
    return r


def max_sharpe_weights(r: np.ndarray, rf: float = RF_P) -> np.ndarray:
    n = r.shape[1]
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        port = r @ w
        mu = port.mean() - rf
        sig = port.std(ddof=0)
        return -(mu / sig) if sig > 1e-8 else 0.0

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return result.x if result.success else w0


def portfolio_returns(weights: np.ndarray, r: np.ndarray) -> np.ndarray:
    rets = []
    prev = np.zeros(weights.shape[1])
    for w, row_r in zip(weights, r):
        turnover = float(np.sum(np.abs(w - prev)))
        rets.append(float(w @ row_r - turnover * COST))
        prev = w
    return np.array(rets)


def metrics(rets: np.ndarray) -> dict:
    cum_ret = float(np.prod(1 + rets) - 1)
    ann_ret = float((1 + cum_ret) ** (PPY / len(rets)) - 1)
    ann_vol = float(rets.std() * np.sqrt(PPY))
    sharpe = float((ann_ret - RF) / ann_vol) if ann_vol > 0 else 0.0
    curve = np.cumprod(1 + rets)
    drawdown = curve / np.maximum.accumulate(curve) - 1
    mdd = float(drawdown.min())
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0
    return {
        "cum_ret": cum_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "curve": curve,
        "drawdown": drawdown,
        "rets": rets,
    }


def build_results():
    data = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
    y_train = data["y_train"]
    y_test = data["y_test"]
    x_test = data["X_test"].astype(np.float32)

    rows_all = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
    train_rows = [r for r in rows_all if r["split"] == "train"]
    test_rows = [r for r in rows_all if r["split"] == "test"]
    prices = {asset: load_prices(f"data/raw/{asset.lower()}_daily.csv") for asset in ASSETS}

    train_r = returns_matrix(train_rows, prices)
    test_r = returns_matrix(test_rows, prices)

    regime_weights = {}
    for regime_id in [0, 1, 2]:
        regime_weights[regime_id] = max_sharpe_weights(train_r[y_train == regime_id])

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = RegimeClassifier(input_size=40, conv_channels=16, lstm_hidden=32).to(device)
    model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(torch.tensor(x_test).to(device)).cpu().numpy()

    n = len(test_rows)
    buy_hold = np.zeros((n, 4))
    buy_hold[:, 0] = 1.0

    equal_weight = np.tile(np.ones(4) / 4, (n, 1))

    sixty_forty = np.zeros((n, 4))
    sixty_forty[:, 0] = 0.60

    agnostic = np.tile(max_sharpe_weights(train_r), (n, 1))

    soft_regime = (
        probs[:, 0:1] * regime_weights[0]
        + probs[:, 1:2] * regime_weights[1]
        + probs[:, 2:3] * regime_weights[2]
    )

    oracle = np.array([regime_weights[int(y)] for y in y_test])

    dl_spy_cash = np.zeros((n, 4))
    dl_spy_cash[:, 0] = probs[:, 2] + 0.5 * probs[:, 1]

    weights = {
        "Buy & Hold": buy_hold,
        "EW 1/N": equal_weight,
        "60/40": sixty_forty,
        "Regime-Agnostic MVO": agnostic,
        "DL Regime SPY/Cash": dl_spy_cash,
        "Regime-MVO": soft_regime,
        "Oracle (HMM labels)": oracle,
    }

    results = {
        name: metrics(portfolio_returns(w, test_r))
        for name, w in weights.items()
    }
    dates = [row["target_date"] for row in test_rows]
    return results, dates


def plot_static_dynamic(results: dict, dates: list[str]):
    order = [
        "Buy & Hold",
        "EW 1/N",
        "60/40",
        "Regime-Agnostic MVO",
        "Regime-MVO",
        "Oracle (HMM labels)",
    ]
    styles = {
        "Buy & Hold": ("--", 1.6),
        "EW 1/N": ("--", 1.8),
        "60/40": ("--", 1.5),
        "Regime-Agnostic MVO": ("--", 2.0),
        "Regime-MVO": ("-", 3.0),
        "Oracle (HMM labels)": (":", 2.4),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True)
    fig.suptitle(
        "Static Benchmarks vs Dynamic Regime-MVO Backtest\n"
        "Regime-MVO does not maximize return, but it reduces drawdown",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(len(dates) + 1)
    for name in order:
        m = results[name]
        curve = np.concatenate([[1.0], m["curve"]])
        drawdown = np.concatenate([[0.0], m["drawdown"]])
        linestyle, linewidth = styles[name]
        label = name
        if name == "Regime-MVO":
            label = "Regime-MVO (dynamic)"
        elif name == "Regime-Agnostic MVO":
            label = "Regime-Agnostic MVO (static)"
        ax1.plot(
            x,
            (curve - 1) * 100,
            color=COLORS[name],
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        ax2.plot(
            x,
            drawdown * 100,
            color=COLORS[name],
            linestyle=linestyle,
            linewidth=linewidth,
        )

    ax1.axhline(0, color="black", linewidth=0.6)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax1.set_ylabel("Cumulative Return (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Rebalancing Period (5-day intervals)")
    ax1.set_title("(a) Cumulative return")
    ax2.set_title("(b) Drawdown path")

    mdd_agnostic = abs(results["Regime-Agnostic MVO"]["mdd"]) * 100
    mdd_ours = abs(results["Regime-MVO"]["mdd"]) * 100
    cum_agnostic = results["Regime-Agnostic MVO"]["cum_ret"] * 100
    cum_ours = results["Regime-MVO"]["cum_ret"] * 100
    note = (
        f"Return trade-off: Agnostic MVO {cum_agnostic:.1f}% vs Regime-MVO {cum_ours:.1f}%\n"
        f"Drawdown improvement: MDD {mdd_agnostic:.1f}% -> {mdd_ours:.1f}% "
        f"({mdd_agnostic - mdd_ours:.1f}pp)"
    )
    ax2.text(
        0.02,
        0.08,
        note,
        transform=ax2.transAxes,
        fontsize=9.5,
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#AAB7B8", alpha=0.92),
    )

    tick_idx = list(range(0, len(dates) + 1, 20))
    tick_lbl = ["Start"] + [dates[i - 1][:7] for i in tick_idx[1:] if i <= len(dates)]
    ax2.set_xticks(tick_idx[:len(tick_lbl)])
    ax2.set_xticklabels(tick_lbl, rotation=30, fontsize=8)
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT_DIR / "fig03_static_dynamic_backtest.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def plot_main_metrics(results: dict):
    order = [
        "Buy & Hold",
        "EW 1/N",
        "60/40",
        "Regime-Agnostic MVO",
        "Regime-MVO",
        "Oracle (HMM labels)",
    ]
    labels = [
        "Buy &\nHold",
        "EW\n1/N",
        "60/40",
        "Agnostic\nMVO",
        "Regime-\nMVO\n(dynamic)",
        "Oracle\n(HMM labels)",
    ]

    cum = [results[n]["cum_ret"] * 100 for n in order]
    mdd = [abs(results[n]["mdd"]) * 100 for n in order]
    calmar = [results[n]["calmar"] for n in order]
    colors = [COLORS[n] for n in order]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    fig.suptitle(
        "Static vs Dynamic Portfolio Results  (Test: 2024.04 ~ 2026.05)\n"
        "Regime-MVO improves downside risk, not cumulative return",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )

    panels = [
        (cum, "Cumulative Return (%)\n(higher = better)", True, "{:.1f}%"),
        (mdd, "Max Drawdown (%)\n(lower = better)", False, "{:.1f}%"),
        (calmar, "Calmar Ratio\n(higher = better)", True, "{:.2f}"),
    ]

    for ax, (values, title, higher, fmt) in zip(axes, panels):
        bars = ax.bar(x, values, color=colors, alpha=0.88, width=0.64)
        best = max(values) if higher else min(values)
        for bar, val, name in zip(bars, values, order):
            if abs(val - best) < 1e-8:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2.8)
            if name == "Regime-MVO":
                bar.set_edgecolor("#1B2631")
                bar.set_linewidth(max(bar.get_linewidth(), 1.8))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=COLORS[name],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10.5, fontweight="bold")

    improvement = mdd[3] - mdd[4]
    axes[1].annotate(
        f"{improvement:.1f}pp MDD reduction\nvs agnostic MVO",
        xy=(4, mdd[4]),
        xytext=(3.25, max(mdd[3], mdd[4]) + 4.0),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.8),
        fontsize=9,
        color="#2C3E50",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#2C3E50", alpha=0.9),
    )

    handles = [
        mpatches.Patch(color=COLORS["Regime-MVO"], label="Dynamic Regime-MVO"),
        mpatches.Patch(color=COLORS["Regime-Agnostic MVO"], label="Static agnostic MVO"),
        mpatches.Patch(color=COLORS["EW 1/N"], label="Static EW 1/N"),
        mpatches.Patch(color=COLORS["Oracle (HMM labels)"], label="Pseudo-label upper bound"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = OUT_DIR / "fig03_main_result.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def main():
    results, dates = build_results()
    plot_static_dynamic(results, dates)
    plot_main_metrics(results)


if __name__ == "__main__":
    main()

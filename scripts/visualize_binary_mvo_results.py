"""
Generate final presentation figures for the Binary Soft Label + 2-Regime MVO result.

Outputs:
  outputs/figures/final/fig03_static_dynamic_backtest.png
  outputs/figures/final/fig03_main_result.png
  outputs/figures/final/fig04_classification_performance.png
  outputs/figures/final/fig05_confusion_matrix.png
  outputs/figures/final/fig07_ablation.png
  outputs/figures/final/fig09_binary_mvo_weights.png
"""

from __future__ import annotations

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
from train import RegimeClassifier  # noqa: E402
from regime_portfolio_policy import ASSETS, get_period_return  # noqa: E402


OUT_DIR = Path("outputs/figures/final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST = 0.001
RF = 0.05
PPY = 252 / 5
RF_P = RF / PPY

COLORS = {
    "60/40": "#BFC7CC",
    "Buy & Hold": "#95A5A6",
    "EW 1/N": "#42B978",
    "3-class Regime-MVO cap 40%": "#E67E22",
    "Binary Regime-MVO Soft cap 40%": "#E74C3C",
    "Binary Regime-MVO Soft cap 50%": "#3498DB",
    "3-class Regime-MVO original": "#9B59B6",
    "Unconstrained Binary MVO": "#7F8C8D",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


def load_prices(path: Path) -> dict[str, float]:
    prices = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            prices[row["Date"]] = float(row["Adj Close"])
    return prices


def returns_matrix(rows: list[dict[str, str]], prices: dict[str, dict[str, float]]) -> np.ndarray:
    R = np.zeros((len(rows), len(ASSETS)), dtype=float)
    for i, row in enumerate(rows):
        for j, asset in enumerate(ASSETS):
            R[i, j] = get_period_return(prices[asset], row["input_end_date"], row["target_date"])
    return R


def max_sharpe_weights(R: np.ndarray, max_weight: float) -> np.ndarray:
    n = R.shape[1]
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        port = R @ w
        mu = port.mean() - RF_P
        sig = port.std(ddof=0)
        return float(-(mu / sig)) if sig > 1e-8 else 0.0

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return result.x if result.success else w0


def portfolio_returns(weights: np.ndarray, R: np.ndarray) -> np.ndarray:
    rets = []
    prev = np.zeros(weights.shape[1], dtype=float)
    for w, r in zip(weights, R):
        turnover = float(np.sum(np.abs(w - prev)))
        rets.append(float(w @ r - turnover * COST))
        prev = w
    return np.array(rets)


def metrics_from_returns(rets: np.ndarray, name: str) -> dict:
    cum_ret = float(np.prod(1 + rets) - 1)
    ann_ret = float((1 + cum_ret) ** (PPY / len(rets)) - 1)
    ann_vol = float(rets.std() * np.sqrt(PPY))
    sharpe = float((ann_ret - RF) / ann_vol) if ann_vol > 0 else 0.0
    curve = np.cumprod(1 + rets)
    mdd = float((curve / np.maximum.accumulate(curve) - 1).min())
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0
    return {
        "name": name,
        "cum_ret": cum_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "rets": rets,
    }


def predict_probs(model_path: Path, X: np.ndarray, num_classes: int) -> np.ndarray:
    model = RegimeClassifier(
        input_size=X.shape[-1],
        conv_channels=16,
        lstm_hidden=32,
        dropout=0.6,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        return model.predict_proba(torch.tensor(X).to(device)).cpu().numpy()


def build_strategy_results() -> tuple[dict[str, dict], list[str], dict]:
    prices = {asset: load_prices(Path(f"data/raw/{asset.lower()}_daily.csv")) for asset in ASSETS}

    data_3 = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
    rows_3 = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
    train_rows_3 = [r for r in rows_3 if r["split"] == "train"]
    test_rows_3 = [r for r in rows_3 if r["split"] == "test"]
    train_R_3 = returns_matrix(train_rows_3, prices)
    test_R = returns_matrix(test_rows_3, prices)
    dates = [r["target_date"] for r in test_rows_3]

    y_train_3 = data_3["y_train"].astype(np.int64)
    X_test_3 = data_3["X_test"].astype(np.float32)
    probs_3 = predict_probs(Path("outputs/models/best_model.pt"), X_test_3, num_classes=3)

    mvo_3_cap40 = {rid: max_sharpe_weights(train_R_3[y_train_3 == rid], 0.4) for rid in [0, 1, 2]}
    weights_3_cap40 = (
        probs_3[:, 0:1] * mvo_3_cap40[0]
        + probs_3[:, 1:2] * mvo_3_cap40[1]
        + probs_3[:, 2:3] * mvo_3_cap40[2]
    )

    data_bin = np.load("data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz", allow_pickle=True)
    rows_bin = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_index.csv")))
    train_rows_bin = [r for r in rows_bin if r["split"] == "train"]
    test_rows_bin = [r for r in rows_bin if r["split"] == "test"]
    train_R_bin = returns_matrix(train_rows_bin, prices)
    test_R_bin = returns_matrix(test_rows_bin, prices)
    y_train_bin = data_bin["y_train"].astype(np.int64)
    X_test_bin = data_bin["X_test"].astype(np.float32)
    probs_bin = predict_probs(Path("outputs/models/best_model_binary_soft_labels.pt"), X_test_bin, num_classes=2)

    mvo_bin_cap40 = {rid: max_sharpe_weights(train_R_bin[y_train_bin == rid], 0.4) for rid in [0, 1]}
    weights_bin_cap40 = probs_bin[:, 0:1] * mvo_bin_cap40[0] + probs_bin[:, 1:2] * mvo_bin_cap40[1]

    ew_w = np.tile(np.ones(len(ASSETS)) / len(ASSETS), (len(test_rows_3), 1))
    bnh_w = np.zeros((len(test_rows_3), len(ASSETS)))
    bnh_w[:, 0] = 1.0
    s6040_w = np.zeros((len(test_rows_3), len(ASSETS)))
    s6040_w[:, 0] = 0.60

    results = {
        "60/40": metrics_from_returns(portfolio_returns(s6040_w, test_R), "60/40"),
        "Buy & Hold": metrics_from_returns(portfolio_returns(bnh_w, test_R), "Buy & Hold"),
        "EW 1/N": metrics_from_returns(portfolio_returns(ew_w, test_R), "EW 1/N"),
        "3-class Regime-MVO cap 40%": metrics_from_returns(
            portfolio_returns(weights_3_cap40, test_R),
            "3-class Regime-MVO cap 40%",
        ),
        "Binary Regime-MVO Soft cap 40%": metrics_from_returns(
            portfolio_returns(weights_bin_cap40, test_R_bin),
            "Binary Regime-MVO Soft cap 40%",
        ),
    }
    weights = {
        "binary_cap40": mvo_bin_cap40,
        "3class_cap40": mvo_3_cap40,
    }
    return results, dates, weights


def percent_label(value: float) -> str:
    return f"{value * 100:.1f}%"


def plot_static_dynamic(results: dict[str, dict], dates: list[str]) -> None:
    selected = [
        "Buy & Hold",
        "EW 1/N",
        "3-class Regime-MVO cap 40%",
        "Binary Regime-MVO Soft cap 40%",
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True)
    fig.suptitle(
        "Final Backtest: Binary Soft Label + Capped 2-Regime MVO\n"
        "Higher return than static benchmarks while keeping drawdown similar to EW 1/N",
        fontsize=15,
        y=0.98,
    )

    x = np.arange(len(dates))
    tick_idx = np.linspace(0, len(dates) - 1, 6, dtype=int)
    for name in selected:
        rets = results[name]["rets"]
        curve = np.cumprod(1 + rets)
        dd = curve / np.maximum.accumulate(curve) - 1
        lw = 3.0 if name == "Binary Regime-MVO Soft cap 40%" else 1.8
        ax1.plot(x, (curve - 1) * 100, label=name, color=COLORS[name], linewidth=lw)
        ax2.plot(x, dd * 100, label=name, color=COLORS[name], linewidth=lw)

    ax1.set_ylabel("Cumulative return (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Test period")
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([dates[i] for i in tick_idx], rotation=0)
    ax1.axhline(0, color="#555555", linewidth=0.8)
    ax2.axhline(0, color="#555555", linewidth=0.8)
    ax1.legend(loc="upper left", ncol=2, fontsize=9)

    final = results["Binary Regime-MVO Soft cap 40%"]
    ew = results["EW 1/N"]
    note = (
        f"Final strategy: CumRet {percent_label(final['cum_ret'])}, "
        f"MDD {percent_label(final['mdd'])}, Calmar {final['calmar']:.2f}  |  "
        f"EW 1/N: CumRet {percent_label(ew['cum_ret'])}, "
        f"MDD {percent_label(ew['mdd'])}, Calmar {ew['calmar']:.2f}"
    )
    fig.text(0.5, 0.015, note, ha="center", fontsize=9.5, color="#2C3E50")
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(OUT_DIR / "fig03_static_dynamic_backtest.png", bbox_inches="tight")
    plt.close(fig)


def plot_main_metrics(results: dict[str, dict]) -> None:
    order = [
        "60/40",
        "Buy & Hold",
        "EW 1/N",
        "3-class Regime-MVO cap 40%",
        "Binary Regime-MVO Soft cap 40%",
    ]
    labels = ["60/40", "Buy &\nHold", "EW\n1/N", "3-class\nMVO cap40", "Binary Soft\nMVO cap40"]
    metrics = [
        ("Cumulative Return (%)\n(higher = better)", "cum_ret", 100, False),
        ("Max Drawdown (%)\n(lower = better)", "mdd", -100, True),
        ("Calmar Ratio\n(higher = better)", "calmar", 1, False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Final Strategy Metric Summary  (Test: 2024.04 ~ 2026.05)",
        fontsize=15,
        y=0.98,
    )

    for ax, (title, key, scale, lower_better) in zip(axes, metrics):
        vals = [results[name][key] * scale for name in order]
        if key == "mdd":
            vals = [abs(v) for v in vals]
        bars = ax.bar(
            np.arange(len(order)),
            vals,
            color=[COLORS[name] for name in order],
            edgecolor=[
                "#F1C40F" if name == "Binary Regime-MVO Soft cap 40%" else "none"
                for name in order
            ],
            linewidth=[
                2.8 if name == "Binary Regime-MVO Soft cap 40%" else 0.0
                for name in order
            ],
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, fontsize=8.5)
        for bar, val in zip(bars, vals):
            label = f"{val:.1f}%" if key != "calmar" else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8.5)
        if lower_better:
            ax.text(0.5, 0.91, "lower is better", transform=ax.transAxes, ha="center", fontsize=8, color="#555555")

    handles = [
        mpatches.Patch(color=COLORS["Binary Regime-MVO Soft cap 40%"], label="Final: Binary Soft MVO cap 40%"),
        mpatches.Patch(color=COLORS["EW 1/N"], label="EW 1/N benchmark"),
        mpatches.Patch(color=COLORS["3-class Regime-MVO cap 40%"], label="3-class capped MVO"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    plt.savefig(OUT_DIR / "fig03_main_result.png", bbox_inches="tight")
    plt.close(fig)


def plot_classification() -> None:
    rows = [
        ("LR baseline", 61.4, 32.6),
        ("RF baseline", 66.3, 53.5),
        ("3-class hard label", 51.9, 60.5),
        ("Binary hard label", 70.2, 58.1),
        ("Binary soft label", 72.4, 67.4),
    ]
    labels = [r[0] for r in rows]
    bal_acc = [r[1] for r in rows]
    bear_rec = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.barh(y - 0.18, bal_acc, height=0.34, label="Balanced Accuracy", color="#5DADE2")
    ax.barh(y + 0.18, bear_rec, height=0.34, label="Bear Recall", color="#E74C3C")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Score (%)")
    ax.set_title("Classification Improvement: Binary Soft Labels Improve Bear Detection")
    ax.legend(loc="lower right")

    for yi, value in zip(y - 0.18, bal_acc):
        ax.text(value + 1.0, yi, f"{value:.1f}%", va="center", fontsize=9)
    for yi, value in zip(y + 0.18, bear_rec):
        ax.text(value + 1.0, yi, f"{value:.1f}%", va="center", fontsize=9)

    ax.axvline(50, color="#888888", linestyle="--", linewidth=0.9)
    fig.text(
        0.5,
        0.02,
        "Binary soft label increases Bear Recall from 58.1% to 67.4%, then feeds probabilities into 2-Regime MVO.",
        ha="center",
        fontsize=9.5,
        color="#2C3E50",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(OUT_DIR / "fig04_classification_performance.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix() -> None:
    history = json.loads(Path("outputs/results/train_history_binary_soft_labels.json").read_text(encoding="utf-8"))
    cm = np.array(history["test_classification"]["confusion_matrix"], dtype=int)
    labels = ["Non-Bear", "Bear"]
    recalls = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Binary Soft Label Confusion Matrix\n(Test split)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(labels)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15, color="#1B2631")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=14)
    fig.text(
        0.5,
        0.035,
        f"Non-Bear Recall {recalls[0]:.1%}  |  Bear Recall {recalls[1]:.1%}",
        ha="center",
        fontsize=10,
        color="#2C3E50",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(OUT_DIR / "fig05_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(results: dict[str, dict]) -> None:
    rows = [
        ("Buy & Hold", "Static SPY benchmark", results["Buy & Hold"]),
        ("EW 1/N", "Static diversified benchmark", results["EW 1/N"]),
        ("3-class Regime-MVO cap 40%", "3-class MVO + concentration cap", results["3-class Regime-MVO cap 40%"]),
        ("Binary Regime-MVO Soft cap 40%", "Binary soft labels + 2-Regime MVO + cap", results["Binary Regime-MVO Soft cap 40%"]),
    ]
    fig = plt.figure(figsize=(14, 8.2))
    fig.suptitle(
        "Ablation Summary: Binary Soft Labels + Capped MVO Give the Best Balanced Result",
        fontsize=14,
        y=0.98,
    )

    ax_table = fig.add_axes([0.04, 0.56, 0.92, 0.34])
    ax_table.axis("off")
    table_rows = []
    for name, change, metric in rows:
        table_rows.append([
            name,
            change,
            percent_label(metric["cum_ret"]),
            f"{metric['sharpe']:.2f}",
            percent_label(metric["mdd"]),
            f"{metric['calmar']:.2f}",
        ])
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["Strategy", "Key idea", "CumRet", "Sharpe", "MDD", "Calmar"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.22, 0.34, 0.11, 0.10, 0.10, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F2D3A")
            cell.set_text_props(color="white", weight="bold")
        elif r == 4:
            cell.set_facecolor("#FDEDEC")
        elif r == 2:
            cell.set_facecolor("#EAF7EF")

    chart_metrics = [
        ("Cumulative Return (%)", "cum_ret", 100),
        ("Sharpe Ratio", "sharpe", 1),
        ("Calmar Ratio", "calmar", 1),
    ]
    for idx, (title, key, scale) in enumerate(chart_metrics):
        ax = fig.add_axes([0.06 + idx * 0.31, 0.10, 0.25, 0.32])
        names = [r[0] for r in rows]
        vals = [r[2][key] * scale for r in rows]
        labels = ["B&H", "EW", "3-class\ncap40", "Binary Soft\ncap40"]
        bars = ax.bar(range(len(rows)), vals, color=[COLORS[n] for n in names])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8.5)
        for bar, value in zip(bars, vals):
            label = f"{value:.1f}%" if key == "cum_ret" else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8.5)

    plt.savefig(OUT_DIR / "fig07_ablation.png", bbox_inches="tight")
    plt.close(fig)


def plot_binary_weights(weights: dict) -> None:
    result_paths = {
        "100% cap": Path("outputs/results/backtest_binary_soft_mvo_results.json"),
        "50% cap": Path("outputs/results/backtest_binary_soft_mvo_cap50_results.json"),
        "40% cap": Path("outputs/results/backtest_binary_soft_mvo_cap40_results.json"),
    }
    regimes = ["Non-Bear", "Bear"]
    caps = list(result_paths.keys())
    data = {}
    for cap, path in result_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        data[cap] = payload["regime_weights"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    asset_colors = {"SPY": "#34495E", "QQQ": "#3498DB", "GLD": "#F1C40F", "TLT": "#8E44AD"}
    for ax, regime in zip(axes, regimes):
        bottom = np.zeros(len(caps))
        for asset in ASSETS:
            vals = [data[cap][regime][asset] * 100 for cap in caps]
            ax.bar(caps, vals, bottom=bottom, label=asset, color=asset_colors[asset])
            bottom += vals
        ax.set_title(f"{regime} MVO weights")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Weight (%)")

    fig.suptitle("MVO Weight Cap Prevents Extreme Binary Regime Portfolios", fontsize=14)
    handles = [mpatches.Patch(color=asset_colors[a], label=a) for a in ASSETS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig(OUT_DIR / "fig09_binary_mvo_weights.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    results, dates, weights = build_strategy_results()
    plot_static_dynamic(results, dates)
    plot_main_metrics(results)
    plot_classification()
    plot_confusion_matrix()
    plot_ablation(results)
    plot_binary_weights(weights)

    print("Saved updated final figures:")
    for name in [
        "fig03_static_dynamic_backtest.png",
        "fig03_main_result.png",
        "fig04_classification_performance.png",
        "fig05_confusion_matrix.png",
        "fig07_ablation.png",
        "fig09_binary_mvo_weights.png",
    ]:
        print(f"  {OUT_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

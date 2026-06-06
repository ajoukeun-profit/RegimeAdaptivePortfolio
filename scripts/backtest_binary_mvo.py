"""
Binary Regime-MVO backtest.

The binary model outputs [P(Non-Bear), P(Bear)]. This script keeps the main
Regime-MVO story intact by learning two MVO portfolios on the train split:

    Non-Bear MVO: train samples with Neutral/Bull labels collapsed together
    Bear MVO    : train samples labeled Bear

On test, weights are blended with model probabilities:

    w_t = P(Non-Bear) * w_non_bear + P(Bear) * w_bear
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier, make_json_serializable  # noqa: E402
from regime_portfolio_policy import ASSETS, get_period_return  # noqa: E402


device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

COST = 0.001
RF = 0.05
PPY = 252 / 5
RF_P = RF / PPY
REGIME_NAMES = {0: "Non-Bear", 1: "Bear"}


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


def max_sharpe_weights(R: np.ndarray, rf: float = RF_P, max_weight: float = 1.0) -> np.ndarray:
    n = R.shape[1]
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"max_weight={max_weight} is infeasible for {n} assets")
    w0 = np.ones(n) / n

    def neg_sharpe(w: np.ndarray) -> float:
        port = R @ w
        mu = port.mean() - rf
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


def portfolio_metrics(weights: np.ndarray, R: np.ndarray, name: str) -> dict:
    rets = []
    prev = np.zeros(weights.shape[1], dtype=float)
    for w, r in zip(weights, R):
        turnover = float(np.sum(np.abs(w - prev)))
        rets.append(float(w @ r - turnover * COST))
        prev = w
    rets = np.array(rets)
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
    }


def predict_probs(model_path: Path, X: np.ndarray) -> np.ndarray:
    model = RegimeClassifier(
        input_size=X.shape[-1],
        conv_channels=16,
        lstm_hidden=32,
        dropout=0.6,
        num_classes=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        return model.predict_proba(torch.tensor(X).to(device)).cpu().numpy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary Regime-MVO backtest")
    parser.add_argument("--data", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_index.csv"))
    parser.add_argument("--model", type=Path, default=Path("outputs/models/best_model_binary_soft_labels.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/backtest_binary_soft_mvo_results.json"))
    parser.add_argument("--max-weight", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    data = np.load(args.data, allow_pickle=True)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    X_test = data["X_test"].astype(np.float32)

    rows_all = list(csv.DictReader(args.index.open()))
    train_rows = [r for r in rows_all if r["split"] == "train"]
    test_rows = [r for r in rows_all if r["split"] == "test"]
    prices = {a: load_prices(Path(f"data/raw/{a.lower()}_daily.csv")) for a in ASSETS}
    train_R = returns_matrix(train_rows, prices)
    test_R = returns_matrix(test_rows, prices)

    mvo_w = {}
    print("\n[Binary 국면별 MVO 비중]")
    print(f"max_weight cap: {args.max_weight:.1%}")
    print(f"{'Regime':<10} {'samples':>7}  {'SPY':>7} {'QQQ':>7} {'GLD':>7} {'TLT':>7}")
    print("-" * 54)
    for rid in [0, 1]:
        mask = y_train == rid
        mvo_w[rid] = max_sharpe_weights(train_R[mask], max_weight=args.max_weight)
        print(
            f"{REGIME_NAMES[rid]:<10} {int(mask.sum()):>7}  "
            + "  ".join(f"{v:>6.1%}" for v in mvo_w[rid])
        )

    probs = predict_probs(args.model, X_test)
    soft_weights = probs[:, 0:1] * mvo_w[0] + probs[:, 1:2] * mvo_w[1]
    hard_weights = np.array([mvo_w[int(pred)] for pred in probs.argmax(axis=1)])
    oracle_weights = np.array([mvo_w[int(y)] for y in y_test])

    ew_weights = np.tile(np.ones(len(ASSETS)) / len(ASSETS), (len(test_rows), 1))
    bnh_weights = np.zeros((len(test_rows), len(ASSETS)))
    bnh_weights[:, 0] = 1.0
    s6040_weights = np.zeros((len(test_rows), len(ASSETS)))
    s6040_weights[:, 0] = 0.60
    agnostic_single = max_sharpe_weights(train_R, max_weight=args.max_weight)
    agnostic_weights = np.tile(agnostic_single, (len(test_rows), 1))

    results = {
        "Buy & Hold": portfolio_metrics(bnh_weights, test_R, "Buy & Hold"),
        "EW 1/N": portfolio_metrics(ew_weights, test_R, "EW 1/N"),
        "60/40": portfolio_metrics(s6040_weights, test_R, "60/40"),
        "Regime-Agnostic MVO": portfolio_metrics(agnostic_weights, test_R, "Regime-Agnostic MVO"),
        "Binary Regime-MVO Soft": portfolio_metrics(soft_weights, test_R, "Binary Regime-MVO Soft"),
        "Binary Regime-MVO Hard": portfolio_metrics(hard_weights, test_R, "Binary Regime-MVO Hard"),
        "Oracle Binary MVO": portfolio_metrics(oracle_weights, test_R, "Oracle Binary MVO"),
    }

    print(f"\n{'=' * 76}")
    print(f"Binary Regime-MVO Backtest: {test_rows[0]['input_end_date']} ~ {test_rows[-1]['target_date']}")
    print(f"{'=' * 76}")
    print(f"{'Strategy':<28} {'Cum':>7} {'Ann':>7} {'Vol':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7}")
    print("-" * 76)
    for name, m in results.items():
        marker = " <- binary mvo" if "Binary Regime-MVO Soft" in name else ""
        print(
            f"{name:<28} {m['cum_ret']:>6.1%}  {m['ann_ret']:>6.1%}  "
            f"{m['ann_vol']:>6.1%}  {m['sharpe']:>6.2f}  {m['mdd']:>7.1%}  "
            f"{m['calmar']:>6.2f}{marker}"
        )

    save = {
        "regime_weights": {
            REGIME_NAMES[rid]: {asset: float(w) for asset, w in zip(ASSETS, mvo_w[rid])}
            for rid in [0, 1]
        },
        "results": results,
        "_config": make_json_serializable(vars(args)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(save, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

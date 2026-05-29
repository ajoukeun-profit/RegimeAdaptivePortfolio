"""Backtest the final no-lead regime momentum tilt strategy.

Run from project root:
    python3 scripts/backtest_regime_advanced.py

Outputs:
    outputs/results/backtest_regime_momentum_results.json
    outputs/results/backtest_regime_momentum_weights.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier  # noqa: E402
from regime_portfolio_policy import (  # noqa: E402
    ASSET_COLS,
    ASSETS,
    compute_return_seeking_weights,
    get_period_return,
)


device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


DATA_PATH = Path("data/processed/cross_asset_supervised_30d_5d.npz")
INDEX_PATH = Path("data/processed/cross_asset_supervised_30d_5d_index.csv")
MODEL_PATH = Path("outputs/models/best_model.pt")
RAW_DIR = Path("data/raw")

RESULT_PATH = Path("outputs/results/backtest_regime_momentum_results.json")
WEIGHT_PATH = Path("outputs/results/backtest_regime_momentum_weights.csv")

TRANSACTION_COST = 0.001
RISK_FREE_RATE = 0.05
PERIODS_PER_YEAR = 252 / 5
CASH_RETURN_PER_PERIOD = 0.0


def load_adj_close_csv(path: Path) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            prices[row["Date"]] = float(row["Adj Close"])
    return prices


def load_prices() -> Dict[str, Dict[str, float]]:
    return {asset: load_adj_close_csv(RAW_DIR / f"{asset.lower()}_daily.csv") for asset in ASSETS}


def load_test_rows() -> List[dict]:
    with INDEX_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["split"] == "test"]


def load_model_probs() -> np.ndarray:
    data = np.load(DATA_PATH, allow_pickle=True)
    x_test = torch.tensor(data["X_test"].astype(np.float32)).to(device)

    model = RegimeClassifier(input_size=40, conv_channels=16, lstm_hidden=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        probs = model.predict_proba(x_test).cpu().numpy()
    return probs


def compute_metrics(
    name: str,
    port_rets: Sequence[float],
    weights: np.ndarray,
    turnovers: Sequence[float],
) -> dict:
    r = np.asarray(port_rets, dtype=float)
    if r.size == 0:
        raise ValueError("empty return series")

    cum_ret = float(np.prod(1.0 + r) - 1.0)
    ann_ret = float((1.0 + cum_ret) ** (PERIODS_PER_YEAR / len(r)) - 1.0)
    ann_vol = float(r.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR))
    sharpe = float((ann_ret - RISK_FREE_RATE) / ann_vol) if ann_vol > 0 else 0.0

    curve = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(curve)
    drawdown = curve / running_max - 1.0
    mdd = float(drawdown.min())
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0

    return {
        "name": name,
        "cum_ret": cum_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "avg_turnover": float(np.mean(turnovers)) if len(turnovers) else 0.0,
        "total_turnover": float(np.sum(turnovers)) if len(turnovers) else 0.0,
        "avg_weight": {col: float(weights[:, i].mean()) for i, col in enumerate(ASSET_COLS)},
    }


def strategy_returns(
    weights: np.ndarray,
    prices: Mapping[str, Mapping[str, float]],
    test_rows: Sequence[dict],
) -> tuple[np.ndarray, np.ndarray]:
    rets = []
    turnovers = []
    prev = np.zeros(weights.shape[1], dtype=float)

    for t, row in enumerate(test_rows):
        w = weights[t]
        asset_rets = np.array(
            [get_period_return(prices[a], row["input_end_date"], row["target_date"]) for a in ASSETS]
            + [CASH_RETURN_PER_PERIOD],
            dtype=float,
        )
        turnover = float(np.sum(np.abs(w[:4] - prev[:4])))
        rets.append(float(w @ asset_rets - turnover * TRANSACTION_COST))
        turnovers.append(turnover)
        prev = w

    return np.asarray(rets, dtype=float), np.asarray(turnovers, dtype=float)


def build_current_model_weights(probs: np.ndarray) -> np.ndarray:
    """Existing project rule: SPY weight = p_bull + 0.5*p_neutral, rest cash."""
    w_spy = np.clip(probs[:, 2] + 0.5 * probs[:, 1], 0.0, 1.0)
    out = np.zeros((len(w_spy), len(ASSET_COLS)), dtype=float)
    out[:, 0] = w_spy
    out[:, 4] = 1.0 - w_spy
    return out


def build_static_weights(test_len: int, weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.repeat(w[None, :], test_len, axis=0)


def build_return_seeking_weights(
    probs: np.ndarray,
    prices: Mapping[str, Mapping[str, float]],
    test_rows: Sequence[dict],
) -> np.ndarray:
    weights = []
    prev = None

    for p, row in zip(probs, test_rows):
        w = compute_return_seeking_weights(
            p,
            prices,
            asof_date=row["input_end_date"],
            prev_w=prev,
        )
        weights.append(w)
        prev = w

    return np.asarray(weights, dtype=float)


def save_weights(path: Path, test_rows: Sequence[dict], probs: np.ndarray, weights: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "p_bear", "p_neutral", "p_bull", *ASSET_COLS])
        for row, p, w in zip(test_rows, probs, weights):
            writer.writerow([row["input_end_date"], *[f"{x:.8f}" for x in p], *[f"{x:.8f}" for x in w]])


def main() -> None:
    prices = load_prices()
    test_rows = load_test_rows()
    probs = load_model_probs()

    if len(test_rows) != len(probs):
        raise RuntimeError(f"test rows ({len(test_rows)}) and model probs ({len(probs)}) mismatch")

    strategies = {
        "Buy & Hold SPY": build_static_weights(len(test_rows), [1.0, 0.0, 0.0, 0.0, 0.0]),
        "60/40 SPY/Cash": build_static_weights(len(test_rows), [0.60, 0.0, 0.0, 0.0, 0.40]),
        "60/40 SPY/TLT": build_static_weights(len(test_rows), [0.60, 0.0, 0.0, 0.40, 0.0]),
        "Current Conv1D+LSTM SPY/Cash": build_current_model_weights(probs),
        "Regime Momentum Tilt": build_return_seeking_weights(probs, prices, test_rows),
    }

    results = {}
    for name, weights in strategies.items():
        r, turnover = strategy_returns(weights, prices, test_rows)
        results[name] = compute_metrics(name, r, weights, turnover)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_weights(WEIGHT_PATH, test_rows, probs, strategies["Regime Momentum Tilt"])

    print(f"\nBacktest: {test_rows[0]['input_end_date']} ~ {test_rows[-1]['target_date']}")
    print("=" * 104)
    print(f"{'Strategy':<34} {'CumRet':>9} {'AnnRet':>9} {'AnnVol':>9} {'Sharpe':>8} {'MDD':>9} {'Calmar':>8} {'AvgTurn':>9}")
    print("-" * 104)
    for name, m in results.items():
        print(
            f"{name:<34} "
            f"{m['cum_ret']:>8.1%} "
            f"{m['ann_ret']:>8.1%} "
            f"{m['ann_vol']:>8.1%} "
            f"{m['sharpe']:>8.2f} "
            f"{m['mdd']:>8.1%} "
            f"{m['calmar']:>8.2f} "
            f"{m['avg_turnover']:>8.1%}"
        )
    print("-" * 104)
    print(f"Saved: {RESULT_PATH}")
    print(f"Saved: {WEIGHT_PATH}")


if __name__ == "__main__":
    main()

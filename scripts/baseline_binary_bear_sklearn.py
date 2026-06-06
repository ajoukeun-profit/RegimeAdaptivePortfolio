"""
Scikit-learn baselines for Bear vs Non-Bear classification.

Compares LogisticRegression and RandomForestClassifier against the binary
Conv1D+LSTM experiment using the same compact 30-day summary features.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from regime_portfolio_policy import get_period_return  # noqa: E402


COST = 0.001
RF = 0.05
PPY = 252 / 5


def make_summary_features(x: np.ndarray) -> np.ndarray:
    recent = x[:, -5:, :]
    return np.concatenate(
        [
            x[:, -1, :],
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
            recent.mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def load_prices(path: Path) -> dict[str, float]:
    prices = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            prices[row["Date"]] = float(row["Adj Close"])
    return prices


def metrics_from_returns(port_rets: np.ndarray, name: str) -> dict:
    cum_ret = float(np.prod(1 + port_rets) - 1)
    ann_ret = float((1 + cum_ret) ** (PPY / len(port_rets)) - 1)
    ann_vol = float(port_rets.std() * np.sqrt(PPY))
    sharpe = float((ann_ret - RF) / ann_vol) if ann_vol > 0 else 0.0
    curve = np.cumprod(1 + port_rets)
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


def classification_report_dict(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "non_bear_precision": float(precision[0]),
        "non_bear_recall": float(recall[0]),
        "non_bear_f1": float(f1[0]),
        "bear_precision": float(precision[1]),
        "bear_recall": float(recall[1]),
        "bear_f1": float(f1[1]),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
    }


def spy_cash_backtest(probs: np.ndarray, rows: list[dict[str, str]], threshold: float) -> dict:
    spy_prices = load_prices(Path("data/raw/spy_daily.csv"))
    spy_rets = np.array([
        get_period_return(spy_prices, row["input_end_date"], row["target_date"])
        for row in rows
    ])

    soft_w = probs[:, 0]
    hard_w = np.where(probs[:, 1] >= threshold, 0.0, 1.0)

    def run(weights: np.ndarray, name: str) -> dict:
        prev = np.concatenate([[0.0], weights[:-1]])
        port_rets = weights * spy_rets - np.abs(weights - prev) * COST
        return metrics_from_returns(port_rets, name)

    return {
        "soft_spy_cash": run(soft_w, "Soft SPY/Cash"),
        "hard_spy_cash": run(hard_w, f"Hard SPY/Cash p_bear>={threshold:.2f}"),
    }


def model_probs(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    if list(model.classes_) == [0, 1]:
        return probs
    out = np.zeros((len(x), 2), dtype=float)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = probs[:, col]
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sklearn binary Bear baselines")
    parser.add_argument("--data", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_bear.npz"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_bear_index.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/baseline_binary_bear_sklearn_results.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    data = np.load(args.data, allow_pickle=True)

    x_train = make_summary_features(data["X_train"].astype(np.float32))
    x_valid = make_summary_features(data["X_valid"].astype(np.float32))
    x_test = make_summary_features(data["X_test"].astype(np.float32))
    y_train = data["y_train"].astype(int)
    y_valid = data["y_valid"].astype(int)
    y_test = data["y_test"].astype(int)

    rows_all = list(csv.DictReader(args.index.open()))
    test_rows = [r for r in rows_all if r["split"] == "test"]

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        C=1.0,
                        solver="lbfgs",
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }

    results = {}
    print("\nSklearn Binary Bear Baselines")
    print("=" * 78)
    print(f"{'Model':<22} {'Valid Bal':>9} {'Test Bal':>9} {'Bear Rec':>9} {'Hard MDD':>9} {'Hard Calmar':>11}")
    print("-" * 78)

    for name, model in models.items():
        model.fit(x_train, y_train)
        valid_probs = model_probs(model, x_valid)
        test_probs = model_probs(model, x_test)
        valid_cls = classification_report_dict(y_valid, valid_probs)
        test_cls = classification_report_dict(y_test, test_probs)
        test_bt = spy_cash_backtest(test_probs, test_rows, args.threshold)

        results[name] = {
            "valid_classification": valid_cls,
            "test_classification": test_cls,
            "test_backtest": test_bt,
        }
        hard_bt = test_bt["hard_spy_cash"]
        print(
            f"{name:<22} {valid_cls['balanced_accuracy']:>8.1%} "
            f"{test_cls['balanced_accuracy']:>8.1%} "
            f"{test_cls['bear_recall']:>8.1%} "
            f"{hard_bt['mdd']:>8.1%} "
            f"{hard_bt['calmar']:>10.2f}"
        )

    results["_config"] = {
        "feature_mode": "summary(last, mean, std, min, max, recent5_mean)",
        "threshold": args.threshold,
        "seed": args.seed,
        "sklearn": True,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

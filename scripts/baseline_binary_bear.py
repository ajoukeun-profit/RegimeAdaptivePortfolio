"""
Classical baselines for Bear vs Non-Bear classification.

This script intentionally avoids scikit-learn so it can run in the current
project environment. It compares:

- Logistic Regression (numpy gradient descent)
- Random Forest style bagged CART trees (small numpy implementation)

The target is the binary dataset:
    Non-Bear = 0, Bear = 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from regime_portfolio_policy import ASSETS, get_period_return  # noqa: E402


COST = 0.001
RF = 0.05
PPY = 252 / 5


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def make_summary_features(x: np.ndarray) -> np.ndarray:
    """Convert (n, 30, 40) into compact window summaries."""
    recent = x[:, -5:, :]
    parts = [
        x[:, -1, :],
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        recent.mean(axis=1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def standardize(train: np.ndarray, valid: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (valid - mean) / std, (test - mean) / std


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=2).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(y) / (2.0 * counts)
    return weights[y]


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    n_iter: int = 3000,
    l2: float = 1e-3,
) -> dict[str, np.ndarray]:
    n, d = x.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    sample_w = class_weights(y)
    sample_w = sample_w / sample_w.mean()

    for _ in range(n_iter):
        p = sigmoid(x @ w + b)
        err = (p - y) * sample_w
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b

    return {"w": w, "b": np.array([b])}


def predict_logistic(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    p_bear = sigmoid(x @ model["w"] + float(model["b"][0]))
    return np.column_stack([1.0 - p_bear, p_bear])


@dataclass
class TreeNode:
    proba: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def gini_from_counts(pos: float, total: float) -> float:
    if total <= 0:
        return 0.0
    p = pos / total
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


def best_split(
    x: np.ndarray,
    y: np.ndarray,
    sample_w: np.ndarray,
    feature_ids: np.ndarray,
    rng: np.random.Generator,
    n_thresholds: int,
    min_samples_leaf: int,
) -> tuple[int | None, float | None, float]:
    total_w = float(sample_w.sum())
    parent_pos = float(sample_w[y == 1].sum())
    parent_gini = gini_from_counts(parent_pos, total_w)
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for feature in feature_ids:
        values = x[:, feature]
        unique = np.unique(values)
        if len(unique) <= 1:
            continue
        if len(unique) > n_thresholds:
            qs = rng.choice(np.linspace(0.05, 0.95, 19), size=n_thresholds, replace=False)
            thresholds = np.quantile(values, qs)
        else:
            thresholds = (unique[:-1] + unique[1:]) / 2.0

        for threshold in thresholds:
            left = values <= threshold
            right = ~left
            if int(left.sum()) < min_samples_leaf or int(right.sum()) < min_samples_leaf:
                continue

            left_w = float(sample_w[left].sum())
            right_w = total_w - left_w
            left_pos = float(sample_w[left & (y == 1)].sum())
            right_pos = parent_pos - left_pos
            child_gini = (
                left_w / total_w * gini_from_counts(left_pos, left_w)
                + right_w / total_w * gini_from_counts(right_pos, right_w)
            )
            gain = parent_gini - child_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = int(feature)
                best_threshold = float(threshold)

    return best_feature, best_threshold, best_gain


def build_tree(
    x: np.ndarray,
    y: np.ndarray,
    sample_w: np.ndarray,
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
    max_features: int,
    n_thresholds: int,
    min_samples_leaf: int,
) -> TreeNode:
    weighted_pos = float(sample_w[y == 1].sum())
    total_w = float(sample_w.sum())
    proba = weighted_pos / total_w if total_w > 0 else float(y.mean())
    node = TreeNode(proba=proba)

    if depth >= max_depth or len(y) < 2 * min_samples_leaf or len(np.unique(y)) == 1:
        return node

    n_features = x.shape[1]
    feature_ids = rng.choice(n_features, size=min(max_features, n_features), replace=False)
    feature, threshold, gain = best_split(
        x,
        y,
        sample_w,
        feature_ids,
        rng,
        n_thresholds=n_thresholds,
        min_samples_leaf=min_samples_leaf,
    )
    if feature is None or threshold is None or gain <= 1e-10:
        return node

    mask = x[:, feature] <= threshold
    node.feature = feature
    node.threshold = threshold
    node.left = build_tree(
        x[mask],
        y[mask],
        sample_w[mask],
        rng,
        depth + 1,
        max_depth,
        max_features,
        n_thresholds,
        min_samples_leaf,
    )
    node.right = build_tree(
        x[~mask],
        y[~mask],
        sample_w[~mask],
        rng,
        depth + 1,
        max_depth,
        max_features,
        n_thresholds,
        min_samples_leaf,
    )
    return node


def predict_tree_one(node: TreeNode, row: np.ndarray) -> float:
    while node.feature is not None and node.threshold is not None:
        if row[node.feature] <= node.threshold:
            if node.left is None:
                break
            node = node.left
        else:
            if node.right is None:
                break
            node = node.right
    return node.proba


def predict_tree(node: TreeNode, x: np.ndarray) -> np.ndarray:
    return np.array([predict_tree_one(node, row) for row in x], dtype=float)


def fit_random_forest(
    x: np.ndarray,
    y: np.ndarray,
    n_trees: int = 200,
    max_depth: int = 5,
    min_samples_leaf: int = 8,
    n_thresholds: int = 8,
    seed: int = 42,
) -> list[TreeNode]:
    rng = np.random.default_rng(seed)
    trees = []
    n = len(y)
    max_features = max(1, int(np.sqrt(x.shape[1])))
    base_w = class_weights(y)

    for _ in range(n_trees):
        idx = rng.integers(0, n, size=n)
        tree = build_tree(
            x[idx],
            y[idx],
            base_w[idx],
            rng,
            depth=0,
            max_depth=max_depth,
            max_features=max_features,
            n_thresholds=n_thresholds,
            min_samples_leaf=min_samples_leaf,
        )
        trees.append(tree)
    return trees


def predict_forest(trees: list[TreeNode], x: np.ndarray) -> np.ndarray:
    p_bear = np.mean([predict_tree(tree, x) for tree in trees], axis=0)
    return np.column_stack([1.0 - p_bear, p_bear])


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, preds):
        cm[int(t), int(p)] += 1

    recalls = []
    precisions = []
    f1s = []
    for i in range(2):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    return {
        "accuracy": float((preds == y_true).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "non_bear_precision": precisions[0],
        "non_bear_recall": recalls[0],
        "non_bear_f1": f1s[0],
        "bear_precision": precisions[1],
        "bear_recall": recalls[1],
        "bear_f1": f1s[1],
        "confusion_matrix": cm.tolist(),
    }


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


def spy_cash_backtest(probs: np.ndarray, rows: list[dict[str, str]], threshold: float) -> dict[str, dict]:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classical binary Bear baselines")
    parser.add_argument("--data", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_bear.npz"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_bear_index.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/baseline_binary_bear_results.json"))
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
    x_train, x_valid, x_test = standardize(x_train, x_valid, x_test)

    rows_all = list(csv.DictReader(args.index.open()))
    test_rows = [r for r in rows_all if r["split"] == "test"]

    logreg = fit_logistic_regression(x_train, y_train)
    logreg_valid_probs = predict_logistic(logreg, x_valid)
    logreg_test_probs = predict_logistic(logreg, x_test)

    forest = fit_random_forest(x_train, y_train, seed=args.seed)
    forest_valid_probs = predict_forest(forest, x_valid)
    forest_test_probs = predict_forest(forest, x_test)

    results = {
        "Logistic Regression": {
            "valid_classification": classification_metrics(y_valid, logreg_valid_probs),
            "test_classification": classification_metrics(y_test, logreg_test_probs),
            "test_backtest": spy_cash_backtest(logreg_test_probs, test_rows, args.threshold),
        },
        "Random Forest": {
            "valid_classification": classification_metrics(y_valid, forest_valid_probs),
            "test_classification": classification_metrics(y_test, forest_test_probs),
            "test_backtest": spy_cash_backtest(forest_test_probs, test_rows, args.threshold),
        },
        "_config": {
            "feature_mode": "summary(last, mean, std, min, max, recent5_mean)",
            "threshold": args.threshold,
            "seed": args.seed,
        },
    }

    print("\nClassical Binary Bear Baselines")
    print("=" * 76)
    print(f"{'Model':<22} {'Valid Bal':>9} {'Test Bal':>9} {'Bear Rec':>9} {'Hard MDD':>9} {'Hard Calmar':>11}")
    print("-" * 76)
    for name in ["Logistic Regression", "Random Forest"]:
        test_cls = results[name]["test_classification"]
        valid_cls = results[name]["valid_classification"]
        hard_bt = results[name]["test_backtest"]["hard_spy_cash"]
        print(
            f"{name:<22} {valid_cls['balanced_accuracy']:>8.1%} "
            f"{test_cls['balanced_accuracy']:>8.1%} "
            f"{test_cls['bear_recall']:>8.1%} "
            f"{hard_bt['mdd']:>8.1%} "
            f"{hard_bt['calmar']:>10.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n결과 저장: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


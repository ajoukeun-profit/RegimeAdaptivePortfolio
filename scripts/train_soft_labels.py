"""
Train Conv1D+LSTM with HMM soft labels.

Two experiment modes:

- Soft-label training: target is [prob_bear, prob_neutral, prob_bull]
- Binary soft-label training: target is [prob_non_bear, prob_bear]
- Confidence weighting: additionally weights each sample by max(HMM probability)

The backtest uses the same regime-conditioned MVO construction as
`backtest_mvo.py`, but writes separate outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier, compute_class_metrics, compute_confusion_matrix, make_json_serializable, set_seed  # noqa: E402
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


def load_prices(path: Path) -> dict[str, float]:
    prices = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            prices[row["Date"]] = float(row["Adj Close"])
    return prices


def returns_matrix(rows: list[dict[str, str]], prices: dict[str, dict[str, float]]) -> np.ndarray:
    R = np.zeros((len(rows), len(ASSETS)))
    for i, row in enumerate(rows):
        for j, asset in enumerate(ASSETS):
            R[i, j] = get_period_return(prices[asset], row["input_end_date"], row["target_date"])
    return R


def max_sharpe_weights(R: np.ndarray, rf: float = RF_P) -> np.ndarray:
    n = R.shape[1]
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        port = R @ w
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


def portfolio_metrics(weights: np.ndarray, R: np.ndarray, name: str) -> dict:
    rets = []
    prev = np.zeros(weights.shape[1] if weights.ndim == 2 else 1)
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


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = -(targets * class_weights.unsqueeze(0) * log_probs).sum(dim=1)
    if sample_weights is not None:
        sample_weights = sample_weights / torch.clamp(sample_weights.mean(), min=1e-12)
        per_sample = per_sample * sample_weights
    return per_sample.mean()


def class_weight_from_hard(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(y) / (float(num_classes) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def evaluate(
    model: RegimeClassifier,
    X: np.ndarray,
    y_hard: np.ndarray,
    y_soft: np.ndarray,
    num_classes: int,
) -> dict:
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(torch.tensor(X).to(device)).cpu().numpy()
    preds = probs.argmax(axis=1)
    cm = compute_confusion_matrix(y_hard, preds, num_classes=num_classes)
    precisions, recalls, f1s, macro_f1 = compute_class_metrics(cm)
    hard_acc = float((preds == y_hard).mean())
    bal_acc = float(np.mean(recalls))
    soft_ce = float(-(y_soft * np.log(np.maximum(probs, 1e-12))).sum(axis=1).mean())
    return {
        "accuracy": hard_acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "soft_cross_entropy": soft_ce,
        "recalls": recalls,
        "precisions": precisions,
        "f1s": f1s,
        "confusion_matrix": cm.tolist(),
        "probs": probs,
    }


def train_model(args: argparse.Namespace, data: np.lib.npyio.NpzFile) -> tuple[RegimeClassifier, dict]:
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_train_soft = data["y_train_soft"].astype(np.float32)
    X_valid = data["X_valid"].astype(np.float32)
    y_valid = data["y_valid"].astype(np.int64)
    y_valid_soft = data["y_valid_soft"].astype(np.float32)
    confidence_train = data["confidence_train"].astype(np.float32)
    label_names = get_label_names(data)
    num_classes = len(label_names)
    focus_idx = 1 if num_classes == 2 else min(1, num_classes - 1)
    focus_name = "Bear" if num_classes == 2 else label_names[focus_idx]

    class_weights = class_weight_from_hard(y_train, num_classes)
    model = RegimeClassifier(
        input_size=X_train.shape[-1],
        conv_channels=args.conv_channels,
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.patience // 2),
    )

    if args.confidence_weighting:
        dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train_soft),
            torch.tensor(confidence_train),
        )
    else:
        dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_soft))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    history = {
        "train_loss": [],
        "val_soft_ce": [],
        "val_acc": [],
        "val_bal_acc": [],
        "val_recalls": [],
        "lr": [],
    }
    best_state = None
    best_epoch = 0
    best_score = -float("inf")
    no_improve = 0

    print("Soft-label model config")
    print(f"  data                  : {args.data}")
    print(f"  confidence_weighting  : {args.confidence_weighting}")
    print(f"  label_names           : {label_names}")
    print(f"  class_weights         : {class_weights.detach().cpu().numpy().round(4).tolist()}")
    print(f"  lr                    : {args.lr}")
    print(f"  weight_decay          : {args.weight_decay}")
    print()
    print(
        f"{'Epoch':>5} {'Train Loss':>10} {'Val SoftCE':>10} {'Val Acc':>8} "
        f"{'Bal Acc':>8} {focus_name[:5] + '-Rec':>8} {'LR':>8}"
    )
    print("-" * 78)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            if args.confidence_weighting:
                X_b, y_b_soft, w_b = batch
                w_b = w_b.to(device)
            else:
                X_b, y_b_soft = batch
                w_b = None
            X_b = X_b.to(device)
            y_b_soft = y_b_soft.to(device)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = soft_cross_entropy(logits, y_b_soft, class_weights, w_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(X_b)
            total_count += len(X_b)

        train_loss = total_loss / total_count
        valid_result = evaluate(model, X_valid, y_valid, y_valid_soft, num_classes)
        scheduler.step(valid_result["soft_cross_entropy"])
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["val_soft_ce"].append(float(valid_result["soft_cross_entropy"]))
        history["val_acc"].append(float(valid_result["accuracy"]))
        history["val_bal_acc"].append(float(valid_result["balanced_accuracy"]))
        history["val_recalls"].append([float(x) for x in valid_result["recalls"]])
        history["lr"].append(float(lr))

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"{epoch:>5} {train_loss:>10.4f} {valid_result['soft_cross_entropy']:>10.4f} "
                f"{valid_result['accuracy']:>7.1%} {valid_result['balanced_accuracy']:>7.1%} "
                f"{valid_result['recalls'][focus_idx]:>7.1%} {lr:>8.2e}"
            )

        improved = valid_result["balanced_accuracy"] > best_score + 1e-4
        if improved:
            best_score = float(valid_result["balanced_accuracy"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\n[Early Stop] epoch {epoch}, best_epoch={best_epoch}, best_bal_acc={best_score:.4f}")
                break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")
    model.load_state_dict(best_state)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.model_output)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val balanced accuracy: {best_score:.4f}")
    print(f"모델 저장 완료: {args.model_output}")
    return model, history


def get_label_names(data: np.lib.npyio.NpzFile) -> list[str]:
    if "label_names" not in data.files:
        return ["Bear", "Neutral", "Bull"]
    return [str(x) for x in data["label_names"].tolist()]


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


def spy_cash_metrics(weights: np.ndarray, rets: np.ndarray, name: str) -> dict:
    prev = np.concatenate([[0.0], weights[:-1]])
    turnover = np.abs(weights - prev)
    return metrics_from_returns(weights * rets - turnover * COST, name)


def multi_asset_metrics(
    weights: np.ndarray,
    rows: list[dict[str, str]],
    prices: dict[str, dict[str, float]],
    name: str,
) -> dict:
    rets = []
    prev = np.zeros(weights.shape[1], dtype=float)
    for w, row in zip(weights, rows):
        asset_rets = np.array([
            get_period_return(prices[a], row["input_end_date"], row["target_date"])
            for a in ASSETS
        ])
        turnover = float(np.sum(np.abs(w - prev)))
        rets.append(float(w @ asset_rets - turnover * COST))
        prev = w
    return metrics_from_returns(np.array(rets), name)


def run_backtest(model: RegimeClassifier, data: np.lib.npyio.NpzFile, index_path: Path) -> dict:
    X_test = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)
    label_names = get_label_names(data)

    rows_all = list(csv.DictReader(index_path.open()))
    train_rows = [r for r in rows_all if r["split"] == "train"]
    test_rows = [r for r in rows_all if r["split"] == "test"]
    prices = {a: load_prices(Path(f"data/raw/{a.lower()}_daily.csv")) for a in ASSETS}
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(torch.tensor(X_test).to(device)).cpu().numpy()

    if len(label_names) == 2:
        spy_rets = np.array([
            get_period_return(prices["SPY"], row["input_end_date"], row["target_date"])
            for row in test_rows
        ])
        p_non_bear = probs[:, 0]
        p_bear = probs[:, 1]
        hard_weights = np.where(p_bear >= 0.50, 0.0, 1.0)
        ew_weights = np.tile(np.ones(len(ASSETS)) / len(ASSETS), (len(test_rows), 1))
        return {
            "Buy & Hold": spy_cash_metrics(np.ones(len(test_rows)), spy_rets, "Buy & Hold"),
            "60/40": spy_cash_metrics(np.full(len(test_rows), 0.60), spy_rets, "60/40"),
            "EW 1/N": multi_asset_metrics(ew_weights, test_rows, prices, "EW 1/N"),
            "Soft Binary SPY/Cash": spy_cash_metrics(p_non_bear, spy_rets, "Soft Binary SPY/Cash"),
            "Hard Binary SPY/Cash p_bear>=0.50": spy_cash_metrics(
                hard_weights,
                spy_rets,
                "Hard Binary SPY/Cash p_bear>=0.50",
            ),
        }

    train_R = returns_matrix(train_rows, prices)
    test_R = returns_matrix(test_rows, prices)

    mvo_w = {}
    for rid in [0, 1, 2]:
        mvo_w[rid] = max_sharpe_weights(train_R[y_train == rid])
    print("\n[국면별 MVO 비중]")
    for rid, name in enumerate(label_names):
        print(f"  {name:<8}: " + " ".join(f"{a}={w:.1%}" for a, w in zip(ASSETS, mvo_w[rid])))

    soft_weights = probs[:, 0:1] * mvo_w[0] + probs[:, 1:2] * mvo_w[1] + probs[:, 2:3] * mvo_w[2]
    ew_w = np.tile(np.ones(4) / 4, (len(test_rows), 1))
    bnh_w = np.zeros((len(test_rows), 4))
    bnh_w[:, 0] = 1.0
    s6040_w = np.zeros((len(test_rows), 4))
    s6040_w[:, 0] = 0.60
    lstm_w = np.zeros((len(test_rows), 4))
    lstm_w[:, 0] = probs[:, 2] + 0.5 * probs[:, 1]
    oracle_w = np.array([mvo_w[int(y)] for y in y_test])

    results = {
        "Buy & Hold": portfolio_metrics(bnh_w, test_R, "Buy & Hold"),
        "EW 1/N": portfolio_metrics(ew_w, test_R, "EW 1/N"),
        "60/40": portfolio_metrics(s6040_w, test_R, "60/40"),
        "DL Regime SPY/Cash": portfolio_metrics(lstm_w, test_R, "DL Regime SPY/Cash"),
        "Regime-MVO SoftLabel": portfolio_metrics(soft_weights, test_R, "Regime-MVO SoftLabel"),
        "Oracle (HMM labels)": portfolio_metrics(oracle_w, test_R, "Oracle (HMM labels)"),
    }
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train soft-label regime classifier")
    parser.add_argument("--data", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_index.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("outputs/models/best_model_binary_soft_labels.pt"))
    parser.add_argument("--history-output", type=Path, default=Path("outputs/results/train_history_binary_soft_labels.json"))
    parser.add_argument("--backtest-output", type=Path, default=Path("outputs/results/backtest_binary_soft_labels_results.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--conv-channels", type=int, default=16)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--confidence-weighting", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5, help="Reserved for binary reporting compatibility.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    data = np.load(args.data, allow_pickle=True)
    label_names = get_label_names(data)
    num_classes = len(label_names)
    required = {"y_train_soft", "y_valid_soft", "y_test_soft", "confidence_train"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Missing soft-label arrays in {args.data}: {sorted(missing)}")

    model, history = train_model(args, data)
    test_result = evaluate(
        model,
        data["X_test"].astype(np.float32),
        data["y_test"].astype(np.int64),
        data["y_test_soft"].astype(np.float32),
        num_classes,
    )
    backtest = run_backtest(model, data, args.index)

    print("\n=== Test Results ===")
    print(f"Accuracy         : {test_result['accuracy']:.1%}")
    print(f"Balanced Accuracy: {test_result['balanced_accuracy']:.1%}")
    print(f"Macro F1         : {test_result['macro_f1']:.1%}")
    print(f"Soft CE          : {test_result['soft_cross_entropy']:.4f}")
    for name, recall in zip(label_names, test_result["recalls"]):
        print(f"  Recall {name:<7}: {recall:.1%}")

    print("\n=== Backtest ===")
    for name, m in backtest.items():
        marker = " <- soft" if "SoftLabel" in name else ""
        print(
            f"{name:<24} cum={m['cum_ret']:>6.1%} sharpe={m['sharpe']:>5.2f} "
            f"mdd={m['mdd']:>7.1%} calmar={m['calmar']:>5.2f}{marker}"
        )

    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "history": history,
        "test_classification": {
            k: v for k, v in test_result.items() if k != "probs"
        },
        "backtest": backtest,
        "args": make_json_serializable(vars(args)),
    }
    args.history_output.write_text(json.dumps(save_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    args.backtest_output.parent.mkdir(parents=True, exist_ok=True)
    args.backtest_output.write_text(json.dumps(backtest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n학습 기록 저장 완료: {args.history_output}")
    print(f"백테스트 저장 완료: {args.backtest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

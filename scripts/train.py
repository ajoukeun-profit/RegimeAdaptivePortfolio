"""
Market Regime Classification: Conv1D + LSTM

Features:
- Conv1D + LSTM classifier
- AdamW optimizer
- Early stopping
- Balanced accuracy based best-model selection
- Neutral class boost
- Seed fixing
- JSON save bug fixed for WindowsPath / Path
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

label_names = ["Bear", "Neutral", "Bull"]


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_json_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def load_dataset(path: Path):
    data = np.load(path, allow_pickle=True)

    return (
        data["X_train"].astype(np.float32),
        data["y_train"].astype(np.int64),
        data["X_valid"].astype(np.float32),
        data["y_valid"].astype(np.int64),
        data["X_test"].astype(np.float32),
        data["y_test"].astype(np.int64),
    )


def print_class_distribution(y_train, y_valid, y_test):
    print("Class distribution")
    print("  labels: 0=Bear, 1=Neutral, 2=Bull")
    print(f"  train: {np.bincount(y_train, minlength=3).tolist()}")
    print(f"  valid: {np.bincount(y_valid, minlength=3).tolist()}")
    print(f"  test : {np.bincount(y_test, minlength=3).tolist()}")
    print()


def make_class_weights(y_train, neutral_boost: float = 1.0):
    n_samples = len(y_train)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)

    if np.any(counts == 0):
        raise ValueError(
            f"All classes must appear in train split. Counts: {counts.tolist()}"
        )

    weights = n_samples / (3.0 * counts)

    # class index 1 = Neutral
    weights[1] *= neutral_boost

    return torch.tensor(weights, dtype=torch.float32).to(device)


def compute_confusion_matrix(y_true, y_pred, num_classes: int = 3):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    return cm


def compute_class_recalls(cm):
    recalls = []

    for i in range(cm.shape[0]):
        total = cm[i].sum()
        recall = cm[i, i] / total if total > 0 else 0.0
        recalls.append(float(recall))

    return recalls


def compute_balanced_accuracy(cm):
    recalls = compute_class_recalls(cm)
    return float(np.mean(recalls)), recalls


def compute_class_metrics(cm):
    """Per-class Precision, Recall, F1 + Macro averages."""
    n = cm.shape[0]
    precisions, recalls, f1s = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(float(p))
        recalls.append(float(r))
        f1s.append(float(f1))
    macro_f1 = float(np.mean(f1s))
    return precisions, recalls, f1s, macro_f1


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        pad = kernel_size // 2
        hidden_channels = max(8, out_channels)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),

            nn.Conv1d(hidden_channels, out_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # input:  (batch, seq_len, features)
        # output: (batch, seq_len, conv_channels)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return x


class RegimeClassifier(nn.Module):
    """
    Input:
        x: (batch, 30, input_size)

    Output:
        logits: (batch, 3)
    """

    def __init__(
        self,
        input_size: int,
        conv_channels: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.6,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.conv = ConvBlock(
            in_channels=input_size,
            out_channels=conv_channels,
            kernel_size=3,
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        direction_mul = 2 if bidirectional else 1
        classifier_input_dim = lstm_hidden * direction_mul

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        x = self.conv(x)

        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]

        logits = self.classifier(h)
        return logits

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────
def train(
    X_train,
    y_train,
    X_valid,
    y_valid,
    input_size: int,
    model_output: Path,
    n_epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 10,
    conv_channels: int = 16,
    lstm_hidden: int = 32,
    lstm_layers: int = 1,
    dropout: float = 0.6,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.0,
    neutral_boost: float = 1.5,
    bidirectional: bool = False,
    print_every: int = 5,
    best_metric: str = "val_bal_acc",
    optimizer_name: str = "adamw",
):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = RegimeClassifier(
        input_size=input_size,
        conv_channels=conv_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    class_weights = make_class_weights(
        y_train=y_train,
        neutral_boost=neutral_boost,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, patience // 2),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_bal_acc": [],
        "val_recall_bear": [],
        "val_recall_neutral": [],
        "val_recall_bull": [],
        "lr": [],
    }

    best_state = None
    best_epoch = 0
    no_improve = 0

    best_score = -float("inf")
    best_val_loss = float("inf")

    print("Model config")
    print(f"  input_size      : {input_size}")
    print(f"  conv_channels   : {conv_channels}")
    print(f"  lstm_hidden     : {lstm_hidden}")
    print(f"  lstm_layers     : {lstm_layers}")
    print(f"  bidirectional   : {bidirectional}")
    print(f"  dropout         : {dropout}")
    print(f"  lr              : {lr}")
    print(f"  weight_decay    : {weight_decay}")
    print(f"  label_smoothing : {label_smoothing}")
    print(f"  neutral_boost   : {neutral_boost}")
    print(f"  optimizer       : {optimizer_name}")
    print(f"  class_weights   : {class_weights.detach().cpu().numpy().round(4).tolist()}")
    print(f"  batch_size      : {batch_size}")
    print(f"  max_epochs      : {n_epochs}")
    print(f"  patience        : {patience}")
    print(f"  best_metric     : {best_metric}")
    print()

    print(
        f"{'Epoch':>5}  "
        f"{'Train Loss':>10}  "
        f"{'Val Loss':>9}  "
        f"{'Val Acc':>8}  "
        f"{'Bal Acc':>8}  "
        f"{'N-Rec':>8}  "
        f"{'LR':>8}"
    )
    print("─" * 76)

    for epoch in range(1, n_epochs + 1):
        # ── train ────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()

            logits = model(X_b)
            loss = criterion(logits, y_b)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y_b)

        train_loss = total_loss / len(y_train)

        # ── validation ───────────────────────────────────────────
        model.eval()

        val_loss = 0.0
        correct = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_b, y_b in valid_loader:
                X_b = X_b.to(device)
                y_b = y_b.to(device)

                logits = model(X_b)
                loss = criterion(logits, y_b)
                preds = logits.argmax(1)

                val_loss += loss.item() * len(y_b)
                correct += (preds == y_b).sum().item()

                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(y_b.cpu().numpy().tolist())

        val_loss /= len(y_valid)
        val_acc = correct / len(y_valid)

        val_cm = compute_confusion_matrix(
            y_true=val_targets,
            y_pred=val_preds,
            num_classes=3,
        )

        val_bal_acc, val_recalls = compute_balanced_accuracy(val_cm)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_bal_acc"].append(float(val_bal_acc))
        history["val_recall_bear"].append(float(val_recalls[0]))
        history["val_recall_neutral"].append(float(val_recalls[1]))
        history["val_recall_bull"].append(float(val_recalls[2]))
        history["lr"].append(float(current_lr))

        if epoch % print_every == 0 or epoch == 1:
            print(
                f"{epoch:>5}  "
                f"{train_loss:>10.4f}  "
                f"{val_loss:>9.4f}  "
                f"{val_acc:>7.1%}  "
                f"{val_bal_acc:>7.1%}  "
                f"{val_recalls[1]:>7.1%}  "
                f"{current_lr:>8.2e}"
            )

        # ── best model selection ─────────────────────────────────
        if best_metric == "val_loss":
            improved = val_loss < best_val_loss - 1e-4
            score_for_log = -val_loss
        elif best_metric == "val_acc":
            improved = val_acc > best_score + 1e-4
            score_for_log = val_acc
        elif best_metric == "val_bal_acc":
            improved = val_bal_acc > best_score + 1e-4
            score_for_log = val_bal_acc
        elif best_metric == "val_neutral_recall":
            improved = val_recalls[1] > best_score + 1e-4
            score_for_log = val_recalls[1]
        else:
            raise ValueError(
                "best_metric must be one of: "
                "val_loss, val_acc, val_bal_acc, val_neutral_recall"
            )

        if improved:
            if best_metric == "val_loss":
                best_val_loss = val_loss
                best_score = score_for_log
            else:
                best_score = score_for_log
                best_val_loss = val_loss

            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= patience:
                print()
                print(
                    f"[Early Stop] epoch {epoch}, "
                    f"best_epoch={best_epoch}, "
                    f"best_metric={best_metric}, "
                    f"best_score={best_score:.4f}, "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                break

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)
    model.to(device)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_output)

    print()
    print(f"Best epoch    : {best_epoch}")
    print(f"Best metric   : {best_metric}")
    print(f"Best score    : {best_score:.4f}")
    print(f"Best val_loss : {best_val_loss:.4f}")
    print(f"모델 저장 완료: {model_output}")

    return model, history


# ─────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    model.eval()

    X_t = torch.tensor(X_test).to(device)
    y_t = torch.tensor(y_test).to(device)

    with torch.no_grad():
        probs = model.predict_proba(X_t)
        preds = probs.argmax(dim=-1)

    acc = (preds == y_t).float().mean().item()

    preds_np = preds.cpu().numpy()

    cm = compute_confusion_matrix(
        y_true=y_test,
        y_pred=preds_np,
        num_classes=3,
    )

    precisions, recalls, f1s, macro_f1 = compute_class_metrics(cm)
    bal_acc = float(np.mean(recalls))

    print()
    print("=== Test Results ===")
    print(f"Accuracy         : {acc:.1%}")
    print(f"Balanced Accuracy: {bal_acc:.1%}  (macro Recall)")
    print(f"Macro F1         : {macro_f1:.1%}")
    print()

    print(f"{'':10}", end="")
    for n in label_names:
        print(f"  {n:>7}", end="")
    print("  (predicted)")

    for i, row_name in enumerate(label_names):
        print(f"{row_name:10}", end="")
        for j in range(3):
            print(f"  {cm[i][j]:>7}", end="")
        print()

    print("(actual)")
    print()

    header = f"  {'Class':>7}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(label_names):
        print(f"  {name:>7}  {precisions[i]:>9.1%}  {recalls[i]:>6.1%}  {f1s[i]:>6.1%}")
    print(f"  {'Macro':>7}  {'':>9}  {bal_acc:>6.1%}  {macro_f1:>6.1%}")

    return acc, bal_acc, recalls, preds_np, probs.cpu().numpy(), cm


# ─────────────────────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────────────────────
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train Conv1D + LSTM regime classifier"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/spy_supervised_30d_5d.npz"),
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("outputs/models/best_model.pt"),
    )

    parser.add_argument(
        "--history-output",
        type=Path,
        default=Path("outputs/results/train_history.json"),
    )

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--conv-channels", type=int, default=16)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.6)

    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument(
        "--neutral-boost",
        type=float,
        default=1.5,
        help="Multiply class weight for Neutral class. Neutral index is 1.",
    )

    parser.add_argument(
        "--best-metric",
        type=str,
        default="val_bal_acc",
        choices=[
            "val_loss",
            "val_acc",
            "val_bal_acc",
            "val_neutral_recall",
        ],
        help="Metric used for best model selection.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer to use. adam: Adam, adamw: AdamW (default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)

    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional LSTM. Classifier input becomes lstm_hidden * 2.",
    )

    return parser


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    set_seed(args.seed)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print()

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(args.data)

    input_size = X_train.shape[-1]

    print(f"Dataset: {args.data}")
    print(
        f"Shapes: "
        f"train={X_train.shape}, "
        f"valid={X_valid.shape}, "
        f"test={X_test.shape}"
    )
    print()

    print_class_distribution(y_train, y_valid, y_test)

    model, history = train(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        input_size=input_size,
        model_output=args.model_output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        conv_channels=args.conv_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        neutral_boost=args.neutral_boost,
        bidirectional=args.bidirectional,
        print_every=args.print_every,
        best_metric=args.best_metric,
        optimizer_name=args.optimizer,
    )

    acc, bal_acc, recalls, preds, probs, cm = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    args.history_output.parent.mkdir(parents=True, exist_ok=True)

    save_obj = {
        "history": history,
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bal_acc),
        "test_recall_bear": float(recalls[0]),
        "test_recall_neutral": float(recalls[1]),
        "test_recall_bull": float(recalls[2]),
        "test_confusion_matrix": cm.tolist(),
        "args": make_json_serializable(vars(args)),
    }

    with args.history_output.open("w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2, ensure_ascii=False)

    print()
    print(f"학습 기록 저장 완료: {args.history_output}")

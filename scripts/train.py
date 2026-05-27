"""
Market Regime Classification: Conv1D + LSTM
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 디바이스 ─────────────────────────────────────────────────────
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

label_names = ["Bear", "Neutral", "Bull"]


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


def make_class_weights(y_train):
    n_samples = len(y_train)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"All classes must appear in train split. Counts: {counts.tolist()}")
    return torch.tensor(n_samples / (3 * counts), dtype=torch.float32).to(device)

# ── 3. ConvBlock ─────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32,           kernel_size, padding=pad),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,          out_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return x


# ── 4. RegimeClassifier ──────────────────────────────────────────
class RegimeClassifier(nn.Module):
    """
    입력: (batch, 30, 10)
    출력: (batch, 3)  logits
    """
    def __init__(self, input_size=10, conv_channels=32,
                 lstm_hidden=64, lstm_layers=1, dropout=0.5):
        super().__init__()
        self.conv = ConvBlock(input_size, conv_channels)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.conv(x)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


# ── 5. 학습 루프 ─────────────────────────────────────────────────
def train(
    X_train,
    y_train,
    X_valid,
    y_valid,
    input_size:   int,
    model_output: Path,
    n_epochs:    int   = 200,
    batch_size:  int   = 16,
    lr:          float = 3e-4,
    patience:    int   = 25,    # early stopping patience
):
    # DataLoader: 데이터를 batch 단위로 쪼개서 제공
    # shuffle=True → 매 epoch마다 순서 섞어서 batch 편향 방지 (시간순 분리는 이미 완료됨)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid)),
        batch_size=batch_size, shuffle=False,
    )

    model     = RegimeClassifier(input_size=input_size).to(device)
    class_weights = make_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ReduceLROnPlateau: val_loss가 patience/2 동안 안 줄면 lr을 절반으로
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}")
    print("─" * 52)

    for epoch in range(1, n_epochs + 1):

        # ── train ────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()           # 이전 gradient 초기화
            logits = model(X_b)             # forward
            loss   = criterion(logits, y_b) # loss 계산
            loss.backward()                 # backward: gradient 계산
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient exploding 방지
            optimizer.step()                # 파라미터 업데이트

            total_loss += loss.item() * len(y_b)

        train_loss = total_loss / len(y_train)

        # ── validation ───────────────────────────────────────────
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():               # gradient 계산 끄기 (메모리, 속도)
            for X_b, y_b in valid_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits    = model(X_b)
                val_loss += criterion(logits, y_b).item() * len(y_b)
                correct  += (logits.argmax(1) == y_b).sum().item()

        val_loss /= len(y_valid)
        val_acc   = correct / len(y_valid)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
                  f"{val_acc:>7.1%}  {current_lr:>8.2e}")

        # ── early stopping ───────────────────────────────────────
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Early Stop] epoch {epoch}, best val_loss={best_val_loss:.4f}")
                break

    # best 모델 복원 후 저장
    model.load_state_dict(best_state)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_output)
    print(f"\n모델 저장 완료: {model_output}")

    return model, history


# ── 6. 테스트셋 평가 ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    model.eval()
    X_t = torch.tensor(X_test).to(device)
    y_t = torch.tensor(y_test).to(device)

    with torch.no_grad():
        probs = model.predict_proba(X_t)         # (105, 3)
        preds = probs.argmax(dim=-1)             # (105,)

    acc = (preds == y_t).float().mean().item()

    # confusion matrix (numpy로)
    preds_np = preds.cpu().numpy()
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_test, preds_np):
        cm[t][p] += 1

    print("\n=== Test Results ===")
    print(f"Accuracy: {acc:.1%}")
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

    return acc, preds_np, probs.cpu().numpy()


# ── 실행 ─────────────────────────────────────────────────────────
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train Conv1D+LSTM regime classifier")
    parser.add_argument("--data", type=Path, default=Path("data/processed/spy_supervised_30d_5d.npz"))
    parser.add_argument("--model-output", type=Path, default=Path("outputs/models/best_model.pt"))
    parser.add_argument("--history-output", type=Path, default=Path("outputs/results/train_history.json"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=25)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(f"Device: {device}\n")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(args.data)
    input_size = X_train.shape[-1]
    print(f"Dataset: {args.data}")
    print(f"Shapes: train={X_train.shape}, valid={X_valid.shape}, test={X_test.shape}\n")

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
    )
    acc, preds, probs = evaluate(model, X_test, y_test)

    # history 저장 (나중에 그래프 그릴 때 씀)
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    with args.history_output.open("w", encoding="utf-8") as f:
        json.dump(history, f)

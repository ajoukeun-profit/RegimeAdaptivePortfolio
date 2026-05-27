"""
저장된 모델(.pt)을 불러와 test 셋 평가만 실행하는 스크립트.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from train import (
    RegimeClassifier,
    compute_balanced_accuracy,
    compute_confusion_matrix,
    load_dataset,
    label_names,
)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


def evaluate(model, X_test, y_test):
    model.eval()

    X_t = torch.tensor(X_test).to(device)
    y_t = torch.tensor(y_test).to(device)

    with torch.no_grad():
        probs = model.predict_proba(X_t)
        preds = probs.argmax(dim=-1)

    acc = (preds == y_t).float().mean().item()
    preds_np = preds.cpu().numpy()

    cm = compute_confusion_matrix(y_true=y_test, y_pred=preds_np, num_classes=3)
    bal_acc, recalls = compute_balanced_accuracy(cm)

    print("=== Test Results ===")
    print(f"Accuracy         : {acc:.1%}")
    print(f"Balanced Accuracy: {bal_acc:.1%}")
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

    print("Per-class recall")
    for i, name in enumerate(label_names):
        print(f"  {name:>7}: {recalls[i]:.1%}")

    return acc, bal_acc, recalls


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on test set")
    parser.add_argument("--model", type=Path, required=True, help="저장된 .pt 파일 경로")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/cross_asset_supervised_30d_5d.npz"),
        help="평가할 데이터셋 경로",
    )
    parser.add_argument("--conv-channels", type=int, default=16)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--bidirectional", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Data   : {args.data}")
    print()

    _, _, _, _, X_test, y_test = load_dataset(args.data)
    input_size = X_test.shape[-1]
    print(f"Test set: {X_test.shape}  (input_size={input_size})")
    print()

    model = RegimeClassifier(
        input_size=input_size,
        conv_channels=args.conv_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )

    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    evaluate(model, X_test, y_test)

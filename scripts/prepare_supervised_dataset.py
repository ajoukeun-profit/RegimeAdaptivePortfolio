#!/usr/bin/env python3
"""
Build supervised learning tensors from daily market data and HMM regime labels.

Output target:
    X[t] = previous N trading days of market features ending at a label date
    y[t] = HMM regime label at the next label date

The HMM probabilities are intentionally not used as model input features. They
are labels/diagnostics, and including them in X would leak the label-generation
model into the predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hmm_regime_labeling import LABEL_TO_CODE, load_market_csv, make_features, parse_date


@dataclass
class SplitBounds:
    train_end: int
    valid_end: int


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        if np.isfinite(sample).all():
            out[i] = float(np.mean(sample))
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        if np.isfinite(sample).all():
            out[i] = float(np.std(sample, ddof=1))
    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    alpha = 2.0 / (span + 1.0)
    first = np.flatnonzero(np.isfinite(values))
    if len(first) == 0:
        return out
    start = int(first[0])
    out[start] = values[start]
    for i in range(start + 1, len(values)):
        if not np.isfinite(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    out = np.full(close.shape, np.nan, dtype=float)
    diff = np.diff(close, prepend=np.nan)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = rolling_mean(gain, window)
    avg_loss = rolling_mean(loss, window)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[~np.isfinite(avg_gain) | ~np.isfinite(avg_loss)] = np.nan
    return out / 100.0


def build_daily_feature_matrix(raw_csv: Path) -> Tuple[List[str], List[str], np.ndarray]:
    market = load_market_csv(raw_csv)
    base = make_features(
        market,
        vol_window=20,
        fast_ma=20,
        slow_ma=60,
        drawdown_window=126,
    )

    close = market.close
    open_ = market.open_
    high = market.high
    low = market.low
    volume = market.volume

    close_to_open = np.log(close / open_)
    high_low_range = np.log(high / low)
    volume_log = np.log1p(volume)
    volume_z20 = (volume_log - rolling_mean(volume_log, 20)) / np.maximum(rolling_std(volume_log, 20), 1e-8)
    rsi14 = rsi(close, 14)

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal
    macd_hist_norm = macd_hist / np.maximum(close, 1e-12)

    feature_names = [
        "log_return",
        "return_1d",
        "volatility_20d_ann",
        "ma_gap_20_60",
        "drawdown_126d",
        "close_to_open_log_return",
        "high_low_log_range",
        "volume_zscore_20d",
        "rsi_14",
        "macd_hist_norm",
    ]
    values = np.column_stack(
        [
            base.log_return,
            base.return_1d,
            base.vol_ann,
            base.ma_gap,
            base.drawdown,
            close_to_open,
            high_low_range,
            volume_z20,
            rsi14,
            macd_hist_norm,
        ]
    )
    date_strings = [d.isoformat() for d in market.dates]
    return date_strings, feature_names, values


def load_label_rows(labels_csv: Path) -> List[Dict[str, str]]:
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {labels_csv}")
    required = {"date", "hmm_label", "hmm_label_code"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {labels_csv}: {sorted(missing)}")
    return rows


def chronological_split(n_samples: int, train_ratio: float, valid_ratio: float) -> SplitBounds:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")
    train_end = int(math.floor(n_samples * train_ratio))
    valid_end = int(math.floor(n_samples * (train_ratio + valid_ratio)))
    if train_end <= 0 or valid_end <= train_end or valid_end >= n_samples:
        raise ValueError(f"Invalid split for n_samples={n_samples}")
    return SplitBounds(train_end=train_end, valid_end=valid_end)


def make_samples(
    daily_dates: List[str],
    daily_features: np.ndarray,
    label_rows: List[Dict[str, str]],
    input_window: int,
    target_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    date_to_idx = {d: i for i, d in enumerate(daily_dates)}
    label_rows = sorted(label_rows, key=lambda row: parse_date(row["date"]))

    x_samples: List[np.ndarray] = []
    y_samples: List[int] = []
    meta_rows: List[Dict[str, object]] = []

    last_start = len(label_rows) - target_horizon
    for label_pos in range(last_start):
        current = label_rows[label_pos]
        target = label_rows[label_pos + target_horizon]
        input_end_date = current["date"]
        target_date = target["date"]
        if input_end_date not in date_to_idx:
            continue
        end_idx = date_to_idx[input_end_date]
        start_idx = end_idx - input_window + 1
        if start_idx < 0:
            continue
        x = daily_features[start_idx : end_idx + 1]
        if x.shape[0] != input_window or not np.isfinite(x).all():
            continue

        y = int(target["hmm_label_code"])
        x_samples.append(x)
        y_samples.append(y)
        meta_rows.append(
            {
                "sample_id": len(meta_rows),
                "input_start_date": daily_dates[start_idx],
                "input_end_date": input_end_date,
                "target_date": target_date,
                "current_label": current["hmm_label"],
                "target_label": target["hmm_label"],
                "target_code": y,
            }
        )

    if not x_samples:
        raise ValueError("No supervised samples were created. Check window size and label dates.")
    return np.stack(x_samples), np.array(y_samples, dtype=np.int64), meta_rows


def standardize_by_train(
    x: np.ndarray,
    split: SplitBounds,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = x[: split.train_end]
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0)
    std = train.reshape(-1, train.shape[-1]).std(axis=0, ddof=1)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def add_split_labels(meta_rows: List[Dict[str, object]], split: SplitBounds) -> None:
    for i, row in enumerate(meta_rows):
        if i < split.train_end:
            row["split"] = "train"
        elif i < split.valid_end:
            row["split"] = "valid"
        else:
            row["split"] = "test"


def write_index_csv(path: Path, meta_rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "input_start_date",
        "input_end_date",
        "target_date",
        "current_label",
        "target_label",
        "target_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)


def class_counts(y: np.ndarray) -> Dict[str, int]:
    code_to_label = {code: label for label, code in LABEL_TO_CODE.items()}
    counts = {label: 0 for label in LABEL_TO_CODE}
    for value in y:
        counts[code_to_label[int(value)]] += 1
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare supervised tensors for regime prediction")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/spy_daily.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/spy_hmm_regime_labels_5d.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/spy_supervised_30d_5d.npz"))
    parser.add_argument("--index-output", type=Path, default=Path("data/processed/spy_supervised_30d_5d_index.csv"))
    parser.add_argument("--meta-output", type=Path, default=Path("data/processed/spy_supervised_30d_5d_meta.json"))
    parser.add_argument("--input-window", type=int, default=30)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    daily_dates, feature_names, daily_features = build_daily_feature_matrix(args.raw)
    label_rows = load_label_rows(args.labels)
    x_raw, y, meta_rows = make_samples(
        daily_dates=daily_dates,
        daily_features=daily_features,
        label_rows=label_rows,
        input_window=args.input_window,
        target_horizon=args.target_horizon,
    )
    split = chronological_split(len(y), args.train_ratio, args.valid_ratio)
    x, mean, std = standardize_by_train(x_raw, split)
    add_split_labels(meta_rows, split)

    x_train = x[: split.train_end]
    y_train = y[: split.train_end]
    x_valid = x[split.train_end : split.valid_end]
    y_valid = y[split.train_end : split.valid_end]
    x_test = x[split.valid_end :]
    y_test = y[split.valid_end :]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=x,
        y=y,
        X_raw=x_raw,
        feature_mean=mean,
        feature_std=std,
        X_train=x_train,
        y_train=y_train,
        X_valid=x_valid,
        y_valid=y_valid,
        X_test=x_test,
        y_test=y_test,
        feature_names=np.array(feature_names),
        label_names=np.array(["Bear", "Neutral", "Bull"]),
    )
    write_index_csv(args.index_output, meta_rows)

    summary = {
        "raw_data": str(args.raw),
        "labels": str(args.labels),
        "output": str(args.output),
        "index_output": str(args.index_output),
        "input_window": args.input_window,
        "target_horizon_label_steps": args.target_horizon,
        "n_samples": int(len(y)),
        "feature_names": feature_names,
        "x_shape": list(x.shape),
        "splits": {
            "train": {"samples": int(len(y_train)), "class_counts": class_counts(y_train)},
            "valid": {"samples": int(len(y_valid)), "class_counts": class_counts(y_valid)},
            "test": {"samples": int(len(y_test)), "class_counts": class_counts(y_test)},
        },
        "standardization": "feature-wise z-score using train split only",
        "label_encoding": LABEL_TO_CODE,
        "split_bounds": asdict(split),
    }
    args.meta_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

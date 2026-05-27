#!/usr/bin/env python3
"""
Build one supervised learning dataset from multiple asset-specific HMM labels.

Each asset is sampled from its own (raw CSV, HMM label CSV) pair. The output
keeps the same NPZ keys as prepare_supervised_dataset.py so train.py can load it
without model-side changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hmm_regime_labeling import LABEL_TO_CODE, parse_date
from prepare_supervised_dataset import (
    SplitBounds,
    build_daily_feature_matrix,
    class_counts,
    load_label_rows,
    make_samples,
    standardize_by_train,
)


LABEL_NAMES = np.array(["Bear", "Neutral", "Bull"])


def asset_key(symbol: str) -> str:
    key = symbol.strip().upper()
    if key.endswith(".US"):
        key = key[:-3]
    if not key:
        raise ValueError("Asset symbol cannot be empty")
    return key


def raw_path_for(raw_dir: Path, symbol: str) -> Path:
    return raw_dir / f"{asset_key(symbol).lower()}_daily.csv"


def label_path_for(labels_dir: Path, symbol: str, fit_step: int) -> Path:
    return labels_dir / f"{asset_key(symbol).lower()}_hmm_regime_labels_{fit_step}d.csv"


def write_index_csv(path: Path, meta_rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "asset",
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


def split_by_unique_target_dates(
    meta_rows: List[Dict[str, object]],
    train_ratio: float,
    valid_ratio: float,
) -> SplitBounds:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")

    unique_dates = sorted({str(row["target_date"]) for row in meta_rows}, key=parse_date)
    train_date_end = int(math.floor(len(unique_dates) * train_ratio))
    valid_date_end = int(math.floor(len(unique_dates) * (train_ratio + valid_ratio)))
    if train_date_end <= 0 or valid_date_end <= train_date_end or valid_date_end >= len(unique_dates):
        raise ValueError(f"Invalid date split for {len(unique_dates)} target dates")

    train_dates = set(unique_dates[:train_date_end])
    valid_dates = set(unique_dates[train_date_end:valid_date_end])

    train_end = 0
    valid_end = 0
    for i, row in enumerate(meta_rows):
        target_date = str(row["target_date"])
        if target_date in train_dates:
            row["split"] = "train"
            train_end = i + 1
            valid_end = i + 1
        elif target_date in valid_dates:
            row["split"] = "valid"
            valid_end = i + 1
        else:
            row["split"] = "test"

    if train_end <= 0 or valid_end <= train_end or valid_end >= len(meta_rows):
        raise ValueError(f"Invalid sample split for {len(meta_rows)} samples")
    return SplitBounds(train_end=train_end, valid_end=valid_end)


def build_asset_samples(
    asset: str,
    raw_path: Path,
    labels_path: Path,
    input_window: int,
    target_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]], List[str]]:
    daily_dates, feature_names, daily_features = build_daily_feature_matrix(raw_path)
    label_rows = load_label_rows(labels_path)
    x, y, meta_rows = make_samples(
        daily_dates=daily_dates,
        daily_features=daily_features,
        label_rows=label_rows,
        input_window=input_window,
        target_horizon=target_horizon,
    )
    for row in meta_rows:
        row["asset"] = asset
    return x, y, meta_rows, feature_names


def split_summary(y: np.ndarray, meta_rows: List[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for split_name in ("train", "valid", "test"):
        idx = [i for i, row in enumerate(meta_rows) if row["split"] == split_name]
        split_y = y[idx]
        asset_counts: Dict[str, int] = {}
        for i in idx:
            asset = str(meta_rows[i]["asset"])
            asset_counts[asset] = asset_counts.get(asset, 0) + 1
        summary[split_name] = {
            "samples": int(len(idx)),
            "class_counts": class_counts(split_y),
            "asset_counts": asset_counts,
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare multi-asset supervised tensors")
    parser.add_argument("--assets", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--fit-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/processed/multi_asset_supervised_30d_5d.npz"))
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("data/processed/multi_asset_supervised_30d_5d_index.csv"),
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=Path("data/processed/multi_asset_supervised_30d_5d_meta.json"),
    )
    parser.add_argument("--input-window", type=int, default=30)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    assets = [asset_key(symbol) for symbol in args.assets]

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []
    feature_names: Optional[List[str]] = None
    sources: Dict[str, Dict[str, str]] = {}

    for asset in assets:
        raw_path = raw_path_for(args.raw_dir, asset)
        labels_path = label_path_for(args.labels_dir, asset, args.fit_step)
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw CSV for {asset}: {raw_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing HMM labels for {asset}: {labels_path}")

        x_asset, y_asset, meta_asset, names = build_asset_samples(
            asset=asset,
            raw_path=raw_path,
            labels_path=labels_path,
            input_window=args.input_window,
            target_horizon=args.target_horizon,
        )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Feature schema mismatch for {asset}: {names} != {feature_names}")

        x_parts.append(x_asset)
        y_parts.append(y_asset)
        meta_rows.extend(meta_asset)
        sources[asset] = {"raw_data": str(raw_path), "labels": str(labels_path)}

    x_raw = np.concatenate(x_parts, axis=0)
    y_unsorted = np.concatenate(y_parts, axis=0)
    order = sorted(
        range(len(meta_rows)),
        key=lambda i: (
            parse_date(str(meta_rows[i]["target_date"])),
            str(meta_rows[i]["asset"]),
            parse_date(str(meta_rows[i]["input_end_date"])),
        ),
    )
    x_raw = x_raw[order]
    y = y_unsorted[order]
    meta_rows = [meta_rows[i] for i in order]

    for i, row in enumerate(meta_rows):
        row["sample_id"] = i

    split = split_by_unique_target_dates(meta_rows, args.train_ratio, args.valid_ratio)
    x, mean, std = standardize_by_train(x_raw, split)

    x_train = x[: split.train_end]
    y_train = y[: split.train_end]
    x_valid = x[split.train_end : split.valid_end]
    y_valid = y[split.train_end : split.valid_end]
    x_test = x[split.valid_end :]
    y_test = y[split.valid_end :]

    asset_to_code = {asset: i for i, asset in enumerate(assets)}
    asset_codes = np.array([asset_to_code[str(row["asset"])] for row in meta_rows], dtype=np.int64)

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
        asset_codes=asset_codes,
        asset_names=np.array(assets),
        feature_names=np.array(feature_names),
        label_names=LABEL_NAMES,
    )
    write_index_csv(args.index_output, meta_rows)

    summary = {
        "assets": assets,
        "sources": sources,
        "output": str(args.output),
        "index_output": str(args.index_output),
        "input_window": args.input_window,
        "target_horizon_label_steps": args.target_horizon,
        "n_samples": int(len(y)),
        "n_target_dates": int(len({row["target_date"] for row in meta_rows})),
        "feature_names": feature_names,
        "x_shape": list(x.shape),
        "splits": split_summary(y, meta_rows),
        "standardization": "feature-wise z-score using train split only",
        "split_policy": "chronological split by unique target_date; all assets on a date stay in the same split",
        "label_encoding": LABEL_TO_CODE,
        "asset_encoding": asset_to_code,
        "split_bounds": asdict(split),
    }
    args.meta_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
교차 자산 피처 데이터셋 생성 (Option A)

- 입력(X): SPY + QQQ + GLD + TLT 의 30일 피처를 concat → (n_samples, 30, 40)
- 라벨(y): SPY HMM 국면만 사용 → 라벨 충돌 없음

기존 multi-asset 방식과의 차이:
  기존: 자산별로 (자산 피처, 자산 라벨) 쌍 → GLD Bull + SPY Bear 충돌
  이 방식: SPY 라벨 날짜 기준, 4자산 피처를 하나의 입력으로 합침
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare_supervised_dataset import (
    build_daily_feature_matrix,
    chronological_split,
    load_label_rows,
    standardize_by_train,
    add_split_labels,
    write_index_csv,
)
from hmm_regime_labeling import parse_date, LABEL_TO_CODE

BASE = Path(__file__).parent.parent

ASSETS = ["spy", "qqq", "gld", "tlt"]
RAW_PATHS = {a: BASE / f"data/raw/{a}_daily.csv" for a in ASSETS}
SPY_LABELS = BASE / "data/processed/spy_hmm_regime_labels_5d.csv"
OUTPUT_NPZ = BASE / "data/processed/cross_asset_supervised_30d_5d.npz"
OUTPUT_IDX = BASE / "data/processed/cross_asset_supervised_30d_5d_index.csv"
OUTPUT_META = BASE / "data/processed/cross_asset_supervised_30d_5d_meta.json"
INPUT_WINDOW = 30
TARGET_HORIZON = 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cross-asset supervised dataset")
    parser.add_argument("--output", type=Path, default=OUTPUT_NPZ)
    parser.add_argument("--index-output", type=Path, default=OUTPUT_IDX)
    parser.add_argument("--meta-output", type=Path, default=OUTPUT_META)
    parser.add_argument(
        "--binary-bear",
        action="store_true",
        help="Convert labels to Non-Bear=0 vs Bear=1.",
    )
    parser.add_argument(
        "--soft-labels",
        action="store_true",
        help="Save HMM probability vectors as soft targets. With --binary-bear, use [P(Non-Bear), P(Bear)].",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    # 1. 각 자산의 일별 피처 행렬 로드
    print("피처 행렬 로드 중...")
    asset_dates: dict[str, list[str]] = {}
    asset_features: dict[str, np.ndarray] = {}
    feature_names: list[str] = []

    for asset in ASSETS:
        dates, fnames, feats = build_daily_feature_matrix(RAW_PATHS[asset])
        asset_dates[asset] = dates
        asset_features[asset] = feats
        if not feature_names:
            feature_names = fnames
        print(f"  {asset.upper()}: {len(dates)}일, {feats.shape[1]}개 피처")

    # 2. SPY HMM 라벨 로드 (타겟 라벨은 SPY만)
    label_rows = load_label_rows(SPY_LABELS)
    label_rows = sorted(label_rows, key=lambda r: parse_date(r["date"]))
    print(f"\nSPY 라벨: {len(label_rows)}개")

    # 3. 자산별 date→idx 맵
    date_to_idx = {asset: {d: i for i, d in enumerate(asset_dates[asset])} for asset in ASSETS}

    if args.binary_bear:
        label_names = ["Non-Bear", "Bear"]
        label_encoding = {"Non-Bear": 0, "Bear": 1}
        label_description = "SPY HMM Bear vs Non-Bear"
    else:
        label_names = ["Bear", "Neutral", "Bull"]
        label_encoding = LABEL_TO_CODE
        label_description = "SPY HMM regime only"

    # 4. 샘플 생성: SPY 라벨 날짜 기준, 4자산 피처 concat
    x_samples, y_samples, meta_rows = [], [], []
    y_soft_samples, confidence_samples = [], []
    last_start = len(label_rows) - TARGET_HORIZON

    for label_pos in range(last_start):
        current = label_rows[label_pos]
        target = label_rows[label_pos + TARGET_HORIZON]
        input_end_date = current["date"]
        target_date = target["date"]

        # 4자산 모두 해당 날짜에 데이터 있는지 확인
        windows = []
        valid = True
        for asset in ASSETS:
            if input_end_date not in date_to_idx[asset]:
                valid = False
                break
            end_idx = date_to_idx[asset][input_end_date]
            start_idx = end_idx - INPUT_WINDOW + 1
            if start_idx < 0:
                valid = False
                break
            w = asset_features[asset][start_idx: end_idx + 1]
            if w.shape[0] != INPUT_WINDOW or not np.isfinite(w).all():
                valid = False
                break
            windows.append(w)

        if not valid:
            continue

        # (30, 10) × 4 → (30, 40)
        x = np.concatenate(windows, axis=1)
        original_y = int(target["hmm_label_code"])
        if args.binary_bear:
            y = 1 if original_y == LABEL_TO_CODE["Bear"] else 0
            current_label = "Bear" if int(current["hmm_label_code"]) == LABEL_TO_CODE["Bear"] else "Non-Bear"
            target_label = "Bear" if y == 1 else "Non-Bear"
            if args.soft_labels:
                p_bear = float(target["prob_bear"])
                p_non_bear = float(target["prob_neutral"]) + float(target["prob_bull"])
                y_soft = np.array([p_non_bear, p_bear], dtype=np.float32)
                y_soft = y_soft / np.maximum(y_soft.sum(), 1e-12)
                y_soft_samples.append(y_soft)
                confidence_samples.append(float(y_soft.max()))
        else:
            y = original_y
            current_label = current["hmm_label"]
            target_label = target["hmm_label"]
            if args.soft_labels:
                y_soft = np.array(
                    [
                        float(target["prob_bear"]),
                        float(target["prob_neutral"]),
                        float(target["prob_bull"]),
                    ],
                    dtype=np.float32,
                )
                y_soft = y_soft / np.maximum(y_soft.sum(), 1e-12)
                y_soft_samples.append(y_soft)
                confidence_samples.append(float(y_soft.max()))
        x_samples.append(x)
        y_samples.append(y)
        meta_rows.append({
            "sample_id": len(meta_rows),
            "input_end_date": input_end_date,
            "target_date": target_date,
            "current_label": current_label,
            "target_label": target_label,
            "target_code": y,
        })

    X = np.stack(x_samples)   # (n, 30, 40)
    y = np.array(y_samples, dtype=np.int64)
    y_soft = np.stack(y_soft_samples) if args.soft_labels else None
    confidence = np.array(confidence_samples, dtype=np.float32) if args.soft_labels else None
    print(f"\n생성된 샘플: {X.shape}  (라벨: {label_description})")

    # 5. 시간순 분리 + 정규화
    split = chronological_split(len(y), train_ratio=0.70, valid_ratio=0.15)
    X, mean, std = standardize_by_train(X, split)
    add_split_labels(meta_rows, split)

    X_train, y_train = X[:split.train_end], y[:split.train_end]
    X_valid, y_valid = X[split.train_end:split.valid_end], y[split.train_end:split.valid_end]
    X_test,  y_test  = X[split.valid_end:], y[split.valid_end:]
    if args.soft_labels:
        y_train_soft = y_soft[:split.train_end]
        y_valid_soft = y_soft[split.train_end:split.valid_end]
        y_test_soft = y_soft[split.valid_end:]
        confidence_train = confidence[:split.train_end]
        confidence_valid = confidence[split.train_end:split.valid_end]
        confidence_test = confidence[split.valid_end:]

    # 6. 저장
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "label_names": np.array(label_names),
    }
    if args.soft_labels:
        save_arrays.update(
            {
                "y_train_soft": y_train_soft,
                "y_valid_soft": y_valid_soft,
                "y_test_soft": y_test_soft,
                "confidence_train": confidence_train,
                "confidence_valid": confidence_valid,
                "confidence_test": confidence_test,
            }
        )
    np.savez_compressed(args.output, **save_arrays)
    write_index_csv(args.index_output, meta_rows)

    code_to_label = {v: k for k, v in label_encoding.items()}
    def counts(arr):
        return {code_to_label[i]: int((arr == i).sum()) for i in range(len(label_names))}

    meta = {
        "description": f"Cross-asset features (SPY+QQQ+GLD+TLT), {label_description}",
        "assets": ASSETS,
        "label_source": label_description,
        "input_window": INPUT_WINDOW,
        "n_features_per_asset": len(feature_names),
        "total_features": len(feature_names) * len(ASSETS),
        "x_shape": list(X.shape),
        "feature_names": [f"{a}_{f}" for a in ASSETS for f in feature_names],
        "label_names": label_names,
        "target_type": "soft probabilities" if args.soft_labels else "hard labels",
        "confidence_summary": (
            {
                "train_mean": float(confidence_train.mean()),
                "valid_mean": float(confidence_valid.mean()),
                "test_mean": float(confidence_test.mean()),
            }
            if args.soft_labels else None
        ),
        "splits": {
            "train": {"samples": int(len(y_train)), "class_counts": counts(y_train)},
            "valid": {"samples": int(len(y_valid)), "class_counts": counts(y_valid)},
            "test":  {"samples": int(len(y_test)),  "class_counts": counts(y_test)},
        },
        "standardization": "feature-wise z-score using train split only",
        "label_encoding": label_encoding,
    }
    args.meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\n저장 완료: {args.output}")


if __name__ == "__main__":
    main()

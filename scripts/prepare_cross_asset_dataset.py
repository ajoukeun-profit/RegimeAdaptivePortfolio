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


def main():
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

    # 4. 샘플 생성: SPY 라벨 날짜 기준, 4자산 피처 concat
    x_samples, y_samples, meta_rows = [], [], []
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
        y = int(target["hmm_label_code"])
        x_samples.append(x)
        y_samples.append(y)
        meta_rows.append({
            "sample_id": len(meta_rows),
            "input_end_date": input_end_date,
            "target_date": target_date,
            "current_label": current["hmm_label"],
            "target_label": target["hmm_label"],
            "target_code": y,
        })

    X = np.stack(x_samples)   # (n, 30, 40)
    y = np.array(y_samples, dtype=np.int64)
    print(f"\n생성된 샘플: {X.shape}  (라벨: SPY 국면만)")

    # 5. 시간순 분리 + 정규화
    split = chronological_split(len(y), train_ratio=0.70, valid_ratio=0.15)
    X, mean, std = standardize_by_train(X, split)
    add_split_labels(meta_rows, split)

    X_train, y_train = X[:split.train_end], y[:split.train_end]
    X_valid, y_valid = X[split.train_end:split.valid_end], y[split.train_end:split.valid_end]
    X_test,  y_test  = X[split.valid_end:], y[split.valid_end:]

    # 6. 저장
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test,   y_test=y_test,
    )
    write_index_csv(OUTPUT_IDX, meta_rows)

    code_to_label = {v: k for k, v in LABEL_TO_CODE.items()}
    def counts(arr):
        return {code_to_label[i]: int((arr == i).sum()) for i in range(3)}

    meta = {
        "description": "Cross-asset features (SPY+QQQ+GLD+TLT), SPY label only",
        "assets": ASSETS,
        "label_source": "SPY HMM regime only",
        "input_window": INPUT_WINDOW,
        "n_features_per_asset": len(feature_names),
        "total_features": len(feature_names) * len(ASSETS),
        "x_shape": list(X.shape),
        "feature_names": [f"{a}_{f}" for a in ASSETS for f in feature_names],
        "splits": {
            "train": {"samples": int(len(y_train)), "class_counts": counts(y_train)},
            "valid": {"samples": int(len(y_valid)), "class_counts": counts(y_valid)},
            "test":  {"samples": int(len(y_test)),  "class_counts": counts(y_test)},
        },
        "standardization": "feature-wise z-score using train split only",
        "label_encoding": LABEL_TO_CODE,
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"\n저장 완료: {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()

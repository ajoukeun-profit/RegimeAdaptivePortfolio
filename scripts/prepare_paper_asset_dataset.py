#!/usr/bin/env python3
"""
논문 ETF universe 기반 Cross-asset 데이터셋 생성

- 입력(X): 22개 ETF 자산의 30일 피처 concat → (n_samples, 30, 220)
- 라벨(y): SPY HMM 5거래일 국면 (기존과 동일)
- GTIP 제외: 2018년 이후 데이터만 있어 라벨 기간(2012~)의 절반 이상 누락

사용 자산 (22개):
  AGG, CPER, DBC, EEM, EFA, EMB, EWJ, EWY,
  GLD, HYT, IEF, IEV, IJR, IWM, IYR, JKD,
  OIL, RWX, SPY, TIP, TLT, UUP
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

# 제외 자산:
#   GTIP: 2018년 이후 데이터만 있어 라벨 기간(2012~) 절반 이상 누락
#   OIL:  2023-07-21 상장폐지 → 테스트 기간(2024~2026) 전부 누락
ASSETS = [
    "agg", "dbc", "eem", "efa", "emb", "ewj", "ewy",
    "gld", "hyt", "ief", "iev", "ijr", "iwm", "iyr", "jkd",
    "rwx", "spy", "tip", "tlt", "uup",
]
RAW_DIR   = BASE / "data/raw/paper_etfs_daily"
RAW_PATHS = {a: RAW_DIR / f"{a}_daily.csv" for a in ASSETS}

SPY_LABELS  = BASE / "data/processed/spy_hmm_regime_labels_5d_2004.csv"
OUTPUT_NPZ  = BASE / "data/processed/paper20_asset_supervised_30d_5d_2004.npz"
OUTPUT_IDX  = BASE / "data/processed/paper20_asset_supervised_30d_5d_2004_index.csv"
OUTPUT_META = BASE / "data/processed/paper20_asset_supervised_30d_5d_2004_meta.json"
INPUT_WINDOW   = 30
TARGET_HORIZON = 1


def main():
    print(f"사용 자산: {len(ASSETS)}개")
    print(f"피처 수: {len(ASSETS)} × 10 = {len(ASSETS) * 10}\n")

    # 1. 각 자산 일별 피처 행렬 로드
    print("피처 행렬 로드 중...")
    asset_dates: dict[str, list[str]] = {}
    asset_features: dict[str, np.ndarray] = {}
    feature_names: list[str] = []

    for asset in ASSETS:
        path = RAW_PATHS[asset]
        if not path.exists():
            print(f"  [{asset.upper()}] 파일 없음: {path} — 스킵")
            continue
        dates, fnames, feats = build_daily_feature_matrix(path)
        asset_dates[asset] = dates
        asset_features[asset] = feats
        if not feature_names:
            feature_names = fnames
        print(f"  {asset.upper():5s}: {len(dates)}일, {feats.shape[1]}피처  ({dates[0]} ~ {dates[-1]})")

    available_assets = list(asset_features.keys())
    print(f"\n로드 성공: {len(available_assets)}개 자산")

    # 2. SPY HMM 라벨 로드
    label_rows = load_label_rows(SPY_LABELS)
    label_rows = sorted(label_rows, key=lambda r: parse_date(r["date"]))
    print(f"SPY 라벨: {len(label_rows)}개  ({label_rows[0]['date']} ~ {label_rows[-1]['date']})\n")

    date_to_idx = {
        asset: {d: i for i, d in enumerate(asset_dates[asset])}
        for asset in available_assets
    }

    # 3. 샘플 생성
    x_samples, y_samples, meta_rows = [], [], []
    skipped_missing = 0

    for label_pos in range(len(label_rows) - TARGET_HORIZON):
        current = label_rows[label_pos]
        target  = label_rows[label_pos + TARGET_HORIZON]
        input_end_date = current["date"]
        target_date    = target["date"]

        windows = []
        valid = True
        for asset in available_assets:
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
            skipped_missing += 1
            continue

        x = np.concatenate(windows, axis=1)   # (30, n_assets*10)
        y = int(target["hmm_label_code"])
        x_samples.append(x)
        y_samples.append(y)
        meta_rows.append({
            "sample_id":      len(meta_rows),
            "input_end_date": input_end_date,
            "target_date":    target_date,
            "current_label":  current["hmm_label"],
            "target_label":   target["hmm_label"],
            "target_code":    y,
        })

    print(f"생성된 샘플: {len(x_samples)}개  (스킵: {skipped_missing}개)")

    if len(x_samples) == 0:
        print("샘플이 없습니다. 데이터 범위를 확인하세요.")
        return

    X = np.stack(x_samples)
    y = np.array(y_samples, dtype=np.int64)
    print(f"X shape: {X.shape}  (n_samples, {INPUT_WINDOW}, {len(available_assets) * len(feature_names)})")

    # 4. 클래스 분포 확인
    code_to_label = {v: k for k, v in LABEL_TO_CODE.items()}
    print("\n라벨 분포:")
    for code, label in sorted(code_to_label.items()):
        print(f"  {label}: {(y == code).sum()}개 ({(y == code).mean():.1%})")

    # 5. 시간순 분리 + 정규화
    split = chronological_split(len(y), train_ratio=0.70, valid_ratio=0.15)
    X, mean, std = standardize_by_train(X, split)
    add_split_labels(meta_rows, split)

    X_train, y_train = X[:split.train_end], y[:split.train_end]
    X_valid, y_valid = X[split.train_end:split.valid_end], y[split.train_end:split.valid_end]
    X_test,  y_test  = X[split.valid_end:], y[split.valid_end:]

    print(f"\nTrain: {len(y_train)}개 / Valid: {len(y_valid)}개 / Test: {len(y_test)}개")

    # 6. 저장
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test,   y_test=y_test,
    )
    write_index_csv(OUTPUT_IDX, meta_rows)

    def counts(arr):
        return {code_to_label[i]: int((arr == i).sum()) for i in range(3)}

    meta = {
        "description": f"Paper ETF universe ({len(available_assets)} assets, GTIP 제외), SPY label",
        "assets": available_assets,
        "n_assets": len(available_assets),
        "label_source": "SPY HMM 5-day regime",
        "input_window": INPUT_WINDOW,
        "n_features_per_asset": len(feature_names),
        "total_features": len(available_assets) * len(feature_names),
        "x_shape": list(X.shape),
        "feature_names": [f"{a}_{f}" for a in available_assets for f in feature_names],
        "splits": {
            "train": {"samples": int(len(y_train)), "class_counts": counts(y_train)},
            "valid": {"samples": int(len(y_valid)), "class_counts": counts(y_valid)},
            "test":  {"samples": int(len(y_test)),  "class_counts": counts(y_test)},
        },
        "standardization": "feature-wise z-score (train split only)",
        "label_encoding": LABEL_TO_CODE,
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n저장 완료:")
    print(f"  {OUTPUT_NPZ}")
    print(f"  {OUTPUT_IDX}")
    print(f"  {OUTPUT_META}")


if __name__ == "__main__":
    main()

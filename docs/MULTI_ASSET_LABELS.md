# 다자산 HMM 라벨 생성

## 목적

기존 SPY 단일 자산 기준 HMM 라벨은 학습 샘플이 적고 특정 시장 구조에 편향될 수 있다. QQQ, GLD, TLT를 추가해 자산별 원본 시계열 다양성을 늘리고, 같은 HMM 규칙으로 Bear / Neutral / Bull 라벨을 생성한다.

## 생성 명령

```bash
python3 scripts/generate_multi_asset_hmm_labels.py \
  --assets SPY QQQ GLD TLT \
  --start 2010-01-01 --end 2026-05-15
```

기본 설정은 기존 SPY 5거래일 라벨과 맞춘다.

- 504거래일 rolling window
- 3-state Gaussian HMM
- 5거래일 간격 라벨
- smoothing window 5
- 다음 라벨을 `target_label_plus_1_steps`로 저장

## 산출물

| 파일 | 설명 |
|------|------|
| `data/raw/{asset}_daily.csv` | 자산별 OHLCV 원천 데이터 |
| `data/processed/{asset}_hmm_regime_labels_5d.csv` | 자산별 HMM 라벨 |
| `data/processed/multi_asset_hmm_regime_labels_5d.csv` | `asset` 컬럼을 붙인 통합 라벨 |

## 사용 시 주의

- 통합 라벨 CSV는 자산별 라벨을 길게 붙인 진단/추적용 파일이다.
- 현재 `prepare_supervised_dataset.py`는 단일 자산 raw CSV와 단일 라벨 CSV를 받아 샘플을 만든다.
- 따라서 통합 라벨 CSV를 그대로 `--labels`에 넣으면 안 된다. SPY 가격 feature에 QQQ/GLD/TLT 라벨이 섞이는 데이터 오류가 생긴다.
- 다자산 학습 데이터셋은 `prepare_multi_asset_supervised_dataset.py`로 만든다. 이 스크립트는 자산별로 `(raw, labels)` 쌍을 맞춰 샘플을 생성한 뒤, train/valid/test를 target date 기준으로 다시 나눈다.

## 학습 데이터셋 생성

```bash
python3 scripts/prepare_multi_asset_supervised_dataset.py
```

출력:

| 파일 | 설명 |
|------|------|
| `data/processed/multi_asset_supervised_30d_5d.npz` | `train.py`가 바로 읽는 다자산 학습 배열 |
| `data/processed/multi_asset_supervised_30d_5d_index.csv` | 샘플별 asset/date/split 추적용 인덱스 |
| `data/processed/multi_asset_supervised_30d_5d_meta.json` | split, class count, asset count 요약 |

현재 생성 결과는 총 2,792개 샘플이다.

| split | samples | asset 구성 |
|------|---------|------------|
| train | 1,952 | 자산별 488개 |
| valid | 420 | 자산별 105개 |
| test | 420 | 자산별 105개 |

같은 `target_date`의 네 자산 샘플은 같은 split에 들어가도록 처리한다.

## 학습 실행

```bash
python3 scripts/train.py \
  --data data/processed/multi_asset_supervised_30d_5d.npz \
  --model-output outputs/models/best_model_multi_asset.pt \
  --history-output outputs/results/train_history_multi_asset.json
```

## 해석

Bear / Neutral / Bull은 자산별 rolling window 안에서 Sharpe ratio 순위로 매핑된다. GLD의 Bull과 SPY의 Bull은 같은 절대 수익률 수준을 의미하지 않고, 각 자산의 최근 2년 상태 중 상대적으로 Sharpe가 높은 regime을 의미한다.

# HMM Regime Labeling for SPY

SPY 일별 가격 데이터로 rolling Gaussian HMM을 학습해 시장 국면 라벨을 만들고, 이 라벨을 이용해 지도학습용 시계열 데이터셋을 생성하는 프로젝트입니다.

참고 논문은 `jrfm-12-00168-v2.pdf`입니다.

## 구성

```text
.
├── data
│   ├── raw
│   │   └── spy_daily.csv
│   └── processed
│       ├── spy_hmm_regime_labels.csv
│       ├── spy_hmm_regime_labels_5d.csv
│       ├── spy_supervised_30d_5d.npz
│       ├── spy_supervised_30d_5d_index.csv
│       └── spy_supervised_30d_5d_meta.json
├── scripts
│   ├── hmm_regime_labeling.py
│   └── prepare_supervised_dataset.py
└── jrfm-12-00168-v2.pdf
```

## 방법 요약

`scripts/hmm_regime_labeling.py`는 다음 절차로 HMM 라벨을 생성합니다.

1. SPY OHLCV CSV를 읽고 adjusted close 기준으로 가격을 정렬합니다.
2. 일별 feature를 계산합니다.
   - `log_return`
   - `volatility_20d_ann`
   - `ma_gap_20_60`
   - `drawdown_126d`
3. 최근 504개 유효 거래일을 rolling window로 사용합니다.
4. 각 window 안에서 3-state Gaussian HMM을 학습합니다.
5. Viterbi 경로로 hidden state sequence를 추정합니다.
6. 각 state의 수익률 Sharpe를 계산합니다.
7. Sharpe가 가장 높은 state를 `Bull`, 가장 낮은 state를 `Bear`, 나머지를 `Neutral`로 매핑합니다.
8. 현재 window 마지막 날짜의 state를 해당 날짜의 HMM 라벨로 저장합니다.

라벨 인코딩은 다음과 같습니다.

```text
Bear    -> 0
Neutral -> 1
Bull    -> 2
```

## 논문 방식과의 차이

현재 구현은 참고 논문의 핵심 아이디어인 `2년 rolling HMM + state별 Sharpe로 상승/하락 국면 판단`을 따르지만, 완전한 1:1 재현은 아닙니다.

주요 차이는 다음과 같습니다.

- 논문은 월간 리밸런싱과 월간 수익률을 사용합니다.
- 현재 구현은 일별 데이터를 사용하고, `fit-step`으로 라벨 생성 간격을 조절합니다.
- 논문은 adjusted close price/return 중심으로 HMM을 적용합니다.
- 현재 구현은 수익률뿐 아니라 변동성, 이동평균 괴리, drawdown feature를 함께 사용합니다.
- 현재 `Bear`, `Neutral`, `Bull`은 절대적인 시장 국면이 아니라 각 rolling window 안에서의 상대적인 Sharpe ranking입니다.

따라서 `Bull`은 "해당 window에서 Sharpe가 가장 높은 hidden state"로 해석하는 것이 정확합니다. `Bear`도 항상 음의 Sharpe라는 뜻은 아니며, 해당 window에서 가장 낮은 Sharpe state라는 의미입니다.

## 산출물

### `data/processed/spy_hmm_regime_labels.csv`

기본 월간 근사 라벨입니다. 기본값 기준 `fit-step=20`으로 약 한 달마다 HMM을 다시 학습하고 라벨을 생성합니다.

현재 파일 기준:

- 기간: `2012-06-29` ~ `2026-05-15`
- 라벨 수: 176개
- 라벨 분포:
  - `Bull`: 79
  - `Bear`: 55
  - `Neutral`: 42

### `data/processed/spy_hmm_regime_labels_5d.csv`

5거래일 간격 라벨입니다. 딥러닝용 supervised dataset의 기본 라벨 파일로 사용됩니다.

현재 파일 기준:

- 기간: `2012-06-29` ~ `2026-05-15`
- 라벨 수: 699개
- 라벨 분포:
  - `Bull`: 295
  - `Bear`: 236
  - `Neutral`: 168

### `data/processed/spy_supervised_30d_5d.npz`

`spy_hmm_regime_labels_5d.csv`를 target으로 사용해 만든 지도학습용 numpy dataset입니다.

현재 파일 기준:

- `X`: `(698, 30, 10)`
- `y`: `(698,)`
- 입력 window: 30 거래일
- target horizon: 다음 HMM 라벨 row
- split:
  - train: 488 samples
  - valid: 105 samples
  - test: 105 samples

입력 feature는 다음 10개입니다.

```text
log_return
return_1d
volatility_20d_ann
ma_gap_20_60
drawdown_126d
close_to_open_log_return
high_low_log_range
volume_zscore_20d
rsi_14
macd_hist_norm
```

## 재생성 방법

Python 3와 `numpy`가 필요합니다. 스크립트는 `pandas`, `sklearn`, `hmmlearn` 없이 동작하도록 작성되어 있습니다.

### 기본 월간 근사 HMM 라벨 생성

```bash
python3 scripts/hmm_regime_labeling.py \
  --input data/raw/spy_daily.csv \
  --output data/processed/spy_hmm_regime_labels.csv \
  --train-window 504 \
  --fit-step 20 \
  --states 3 \
  --smoothing-window 3 \
  --target-horizon 1
```

### 5거래일 간격 HMM 라벨 생성

```bash
python3 scripts/hmm_regime_labeling.py \
  --input data/raw/spy_daily.csv \
  --output data/processed/spy_hmm_regime_labels_5d.csv \
  --train-window 504 \
  --fit-step 5 \
  --states 3 \
  --smoothing-window 5 \
  --target-horizon 1
```

### 지도학습 데이터셋 생성

```bash
python3 scripts/prepare_supervised_dataset.py \
  --raw data/raw/spy_daily.csv \
  --labels data/processed/spy_hmm_regime_labels_5d.csv \
  --output data/processed/spy_supervised_30d_5d.npz \
  --index-output data/processed/spy_supervised_30d_5d_index.csv \
  --meta-output data/processed/spy_supervised_30d_5d_meta.json \
  --input-window 30 \
  --target-horizon 1
```

## 라벨 CSV 주요 컬럼

- `date`: 라벨 날짜
- `hmm_state`: HMM raw hidden state id
- `hmm_label`: `Bear`, `Neutral`, `Bull`
- `hmm_label_code`: 라벨 정수 코드
- `prob_bear`, `prob_neutral`, `prob_bull`: 현재 시점의 state posterior probability를 라벨 기준으로 합산한 값
- `smooth_prob_*`: 라벨 row 기준 rolling average probability
- `state_mean_daily_return`: 현재 state의 window 내 평균 일별 log return
- `state_ann_return`: 현재 state의 연율화 수익률
- `state_ann_vol`: 현재 state의 연율화 변동성
- `state_sharpe`: 현재 state의 Sharpe
- `state_count`: 현재 rolling window 안에서 현재 state에 속한 관측치 수
- `model_loglik`: HMM log likelihood
- `target_label_plus_1_steps`: 다음 라벨 row의 라벨
- `target_code_plus_1_steps`: 다음 라벨 row의 정수 코드

## 주의사항

- `target_label_plus_1_steps`는 마지막 row에서 비어 있습니다. 미래 라벨이 없기 때문에 정상입니다.
- `smooth_prob_*`는 calendar day가 아니라 라벨 row 기준 smoothing입니다.
- supervised dataset의 입력 feature에는 HMM posterior probability를 넣지 않았습니다. 라벨 생성 모델의 정보를 predictor 입력에 직접 넣는 leakage를 피하기 위한 선택입니다.
- 시간 순서 split을 사용합니다. train/valid/test split 전에 전체 데이터를 shuffle하지 않습니다.

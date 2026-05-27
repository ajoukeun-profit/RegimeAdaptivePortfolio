# 시장 국면 인식 기반 동적 포트폴리오 전략

> 금융 딥러닝 기초 기말 프로젝트  
> 아주대학교 금융공학과

---

## 목차

1. [프로젝트 목표](#1-프로젝트-목표)
2. [전체 파이프라인](#2-전체-파이프라인)
3. [데이터 구조 상세](#3-데이터-구조-상세)
4. [모델 설계 계획](#4-모델-설계-계획)
5. [포트폴리오 전략](#5-포트폴리오-전략)
6. [평가 방법](#6-평가-방법)
7. [진행 현황](#7-진행-현황)

---

## 1. 프로젝트 목표

주가 자체를 예측하는 것이 아니라, **시장이 지금 어떤 국면(Bull/Neutral/Bear)인지** 인식하고, 그에 따라 **포트폴리오 비중을 동적으로 조절**하는 전략을 구성한다.

### 왜 주가 예측이 아닌가?

금융 시장은 Random Walk 특성을 가진다. 내일 주가가 오를지 내릴지 예측하는 것은 노이즈가 너무 커서 실질적 투자 성과로 이어지기 어렵다. 반면 "지금 시장이 상승 국면인가, 하락 국면인가"는 더 안정적인 신호이며, 이를 기반으로 자산배분 비중을 바꾸는 것이 현실적인 전략이다.

### 팀 역할 분담

| 역할 | 담당 | 산출물 |
|------|------|--------|
| HMM 라벨링 | 팀원 | `spy_hmm_regime_labels_5d.csv`, `multi_asset_hmm_regime_labels_5d.csv` |
| 딥러닝 모델 학습 | 나 | `train.py`, 백테스트 코드 |

---

## 2. 전체 파이프라인

```
[Raw Data]
spy_daily.csv (SPY OHLCV, 2010-01-04 ~ 2026-05-15, 4117일)
qqq_daily.csv / gld_daily.csv / tlt_daily.csv (다자산 확장용)
        │
        ▼
[Step 1: HMM 라벨링]  ← 팀원이 완성
hmm_regime_labeling.py
  - 504일(2년) rolling window로 3-state Gaussian HMM 학습
  - 각 state의 Sharpe ratio를 비교해 Bull/Neutral/Bear 매핑
  - 5거래일 간격으로 라벨 생성
        │
        ▼
spy_hmm_regime_labels_5d.csv (699행, 5거래일 간격 라벨)
multi_asset_hmm_regime_labels_5d.csv (asset 컬럼 포함 통합 라벨)
        │
        ▼
[Step 2: 지도학습 데이터셋 생성]  ← 팀원이 완성
prepare_supervised_dataset.py
prepare_multi_asset_supervised_dataset.py (다자산 확장)
  - 30거래일 입력 window 생성
  - 10개 기술적 지표 feature 계산
  - train/valid/test 시간순 분리 + z-score 정규화
        │
        ▼
spy_supervised_30d_5d.npz (딥러닝 입력 데이터)
multi_asset_supervised_30d_5d.npz (SPY/QQQ/GLD/TLT 통합 학습 데이터)
        │
        ▼
[Step 3: 딥러닝 모델 학습]  ← 내가 담당
train.py
  - Conv1D + LSTM 구조
  - 입력: (batch, 30, 10) → 출력: (batch, 3) softmax 확률
        │
        ▼
[Step 4: 포트폴리오 백테스트]
  - 모델 출력 확률로 주식/현금 비중 결정
  - Buy&Hold, 60/40 등 baseline과 비교
```

---

## 3. 데이터 구조 상세

### 3-1. Raw Data: `data/raw/spy_daily.csv`

SPY(S&P 500 ETF)의 일별 OHLCV 데이터.

```
Date        Open    High    Low     Close   Adj Close   Volume
2010-01-04  112.37  113.38  111.51  113.33  84.79       118944600
...
2026-05-15  ...
```

- **기간**: 2010-01-04 ~ 2026-05-15 (약 16년)
- **행 수**: 4,117행 (거래일 기준)

---

### 3-2. HMM 라벨: `data/processed/spy_hmm_regime_labels_5d.csv`

팀원이 생성한 시장 국면 라벨 파일. 5거래일마다 하나의 라벨 행이 있다.

```
date        hmm_label  hmm_label_code  prob_bear  prob_neutral  prob_bull  target_label_plus_1_steps
2012-06-29  Bear       0               0.9991     0.0008        0.0000     Bear
2012-07-09  Bear       0               0.9999     0.0000        0.0000     Bear
...
2026-05-15  Bull       2               0.0000     0.0000        0.9999     (없음)
```

**핵심 컬럼 설명:**

| 컬럼 | 설명 |
|------|------|
| `hmm_label` | 현재 시점의 시장 국면: Bear / Neutral / Bull |
| `hmm_label_code` | 정수 인코딩: Bear=0, Neutral=1, Bull=2 |
| `prob_bear/neutral/bull` | HMM의 state posterior 확률 (세 값의 합 = 1) |
| `target_label_plus_1_steps` | **5거래일 후** 국면 (모델이 예측해야 할 값) |
| `state_sharpe` | 현재 state의 rolling window 내 Sharpe ratio |

> **라벨 해석 주의**: Bull은 "주가가 반드시 오른다"는 뜻이 아니라, 해당 rolling window 안에서 Sharpe ratio가 가장 높은 state를 의미한다.

다자산 확장 라벨은 `data/processed/multi_asset_hmm_regime_labels_5d.csv`에 저장한다. 이 파일은 `asset` 컬럼으로 SPY/QQQ/GLD/TLT를 구분하는 통합 라벨이며, 자산별 파일은 `data/processed/{asset}_hmm_regime_labels_5d.csv` 형식으로 저장한다.

> **사용 주의**: 현재 지도학습 데이터셋 생성 스크립트는 단일 raw CSV와 단일 labels CSV를 입력으로 받는다. 통합 라벨을 그대로 넣으면 SPY feature에 QQQ/GLD/TLT 라벨이 섞일 수 있으므로, 다자산 학습 데이터셋은 자산별 `(raw, labels)` 쌍으로 샘플을 만든 뒤 합쳐야 한다.

다자산 학습용 배열은 `scripts/prepare_multi_asset_supervised_dataset.py`로 생성한다. 출력 파일은 `data/processed/multi_asset_supervised_30d_5d.npz`이며, 같은 target date의 네 자산 샘플이 같은 split에 들어가도록 나눈다.

---

### 3-3. 딥러닝 데이터셋: `data/processed/spy_supervised_30d_5d.npz`

모델 학습에 바로 사용하는 numpy 배열 묶음.

#### 전체 구조

```
X : (698, 30, 10)   ← 입력 시계열
y : (698,)          ← 타겟 라벨 (5거래일 후 국면)
```

#### 하나의 샘플이란?

샘플 0번을 예시로 보면:

```
input_start_date : 2012-05-18
input_end_date   : 2012-06-29   ← X[0]의 마지막 날
target_date      : 2012-07-09   ← y[0]의 날짜 (5거래일 후)

X[0] shape: (30, 10)  ← 30거래일 × 10개 feature
y[0]       : 0        ← Bear (5거래일 후 국면이 Bear였음)
```

즉, **"지난 30거래일의 시장 데이터를 보고 5거래일 후 국면을 예측"** 하는 것이다.

#### 10개 Input Feature 설명

| # | Feature | 설명 | 단위 |
|---|---------|------|------|
| 0 | `log_return` | 로그 수익률 (ln(close_t / close_{t-1})) | 무차원 |
| 1 | `return_1d` | 단순 일별 수익률 | % |
| 2 | `volatility_20d_ann` | 20일 realized volatility (연율화) | % |
| 3 | `ma_gap_20_60` | 20일 MA - 60일 MA 괴리율 | % |
| 4 | `drawdown_126d` | 126거래일(6개월) 최고점 대비 낙폭 | % |
| 5 | `close_to_open_log_return` | 전일 종가→당일 시가 로그 수익률 (gap) | 무차원 |
| 6 | `high_low_log_range` | 당일 고가/저가 로그 범위 (일중 변동성) | 무차원 |
| 7 | `volume_zscore_20d` | 거래량의 20일 z-score | 표준편차 단위 |
| 8 | `rsi_14` | RSI 14일 (0~100 → z-score 변환됨) | 표준화 |
| 9 | `macd_hist_norm` | MACD Histogram (정규화됨) | 표준화 |

> 모든 feature는 **train set 기준 z-score 정규화** 되어있다. valid/test에는 train의 mean/std를 그대로 적용해 미래 정보가 새지 않도록 했다.

#### Train/Valid/Test 분리

```
|──────────────────────── 698 samples ──────────────────────────|
|────── train (488) ────────|── valid (105) ──|── test (105) ──|
 2012-05-18 ~ 약 2020         약 2020 ~ 2022    약 2022 ~ 2026
```

**시간순 분리 (shuffle 없음)**. 미래 데이터가 과거 학습에 들어가는 look-ahead bias를 방지한다.

#### 클래스 분포

| 국면 | 인코딩 | Train | Valid | Test |
|------|--------|-------|-------|------|
| Bear | 0 | 149 (30.5%) | 43 (41.0%) | 43 (41.0%) |
| Neutral | 1 | 128 (26.2%) | 19 (18.1%) | 21 (20.0%) |
| Bull | 2 | 211 (43.2%) | 43 (41.0%) | 41 (39.0%) |

> Valid/Test에서 Neutral 비율이 낮다 (19~21개). 이 기간(2020~2026)은 급락 후 급등 같은 극단적 국면이 많았기 때문으로 해석된다.

---

## 4. 모델 설계 계획

### 구조: Conv1D + LSTM Classifier

```
입력: (batch, 30, 10)
        │
[Conv1D Block]              ← 단기 패턴 추출 (노이즈 제거)
  Conv1d(10→32, kernel=3)
  ReLU + BatchNorm
  Conv1d(32→64, kernel=3)
  ReLU + BatchNorm
        │
(batch, 26, 64)
        │
[LSTM Layer]                ← 시간 의존성 학습
  LSTM(64→128, 2 layers)
  마지막 hidden state만 사용
        │
(batch, 128)
        │
[Classifier Head]           ← 국면 확률 출력
  Linear(128→64) + ReLU + Dropout
  Linear(64→3)
  Softmax
        │
출력: (batch, 3)  ← [p_bear, p_neutral, p_bull]
```

### 왜 Conv1D + LSTM인가?

- **Conv1D**: 금융 시계열의 높은 노이즈를 완화하고, 국소적 패턴(단기 추세, 변동성 클러스터링)을 추출
- **LSTM**: 장기 시간 의존성 포착. "3개월 전 하락 후 회복 패턴" 같은 맥락을 기억

---

## 5. 포트폴리오 전략

모델이 출력한 확률 [p_bear, p_neutral, p_bull]을 이용해 주식 비중을 결정한다.

```
w_stock = p_bull + 0.5 × p_neutral   (상승 + 중립의 절반)
w_cash  = 1 - w_stock
```

- 이산적 전환(Bull이면 100% 주식)이 아닌 **확률 가중합**으로 급격한 비중 변화를 방지
- **5거래일마다 리밸런싱** (라벨 생성 주기와 동일)
- 과도한 신호에 반응하지 않도록 rolling average smoothing 적용 예정

### Baseline 전략 (비교 대상)

| 전략 | 설명 |
|------|------|
| Buy & Hold | SPY 100% 유지 |
| 60/40 | 주식 60% / 현금(or 채권) 40% 고정 |
| 80/20 / 40/60 | 비중 변형 정적 전략 |
| Moving Average Crossover | 단기/장기 MA 교차 신호 기반 규칙 전략 |
| Risk Parity | 변동성 역비례 비중 배분 |

---

## 6. 평가 방법

### 분류 성능

| 지표 | 설명 |
|------|------|
| Accuracy | 전체 정확도 |
| F1-score (macro) | 클래스 불균형 보정 정확도 |
| Confusion Matrix | 어떤 국면을 어떻게 혼동하는지 |

### 투자 성과

| 지표 | 설명 |
|------|------|
| 누적 수익률 | 전체 기간 총 수익 |
| Sharpe Ratio | 수익률 / 변동성 (위험 대비 수익) |
| Max Drawdown | 최대 낙폭 |
| Calmar Ratio | 연율화 수익률 / Max Drawdown |

> 거래 비용(편도 0.1% 가정)을 반영한 성과도 함께 계산한다.

---

## 7. 진행 현황

| 단계 | 상태 | 설명 |
|------|------|------|
| HMM 라벨링 | ✅ 완료 | 팀원 담당 |
| 지도학습 데이터셋 생성 | ✅ 완료 | 팀원 담당 |
| Step 1: 데이터 로드 | ✅ 완료 | `scripts/train.py` |
| Step 2: Conv1D 구현 | ✅ 완료 | `ConvBlock` — 파라미터 7,392개, MPS 동작 확인 |
| Step 3: LSTM 연결 | ✅ 완료 | `RegimeClassifier` — 최종 ~31,300개 파라미터 |
| Step 4: 학습 루프 | ✅ 완료 | Test Accuracy 58.1%, 상세 결과: `TRAINING_RESULTS.md` |
| Step 5: 백테스트 | ✅ 완료 | Calmar 1.30~1.38 (1위), MDD 최저, 상세: `BACKTEST_RESULTS.md` |
| 모델 개선 실험 | ✅ 완료 | 4개 실험, 최종 61.0% (Exp3), 상세: `MODEL_IMPROVEMENT.md` |
| 시각화 | ✅ 완료 | 발표용 그래프 4개 (`data/processed/fig*.png`) |
| 팀원용 요약 | ✅ 완료 | `FOR_TEAMMATES.md` |

---

## 참고 문헌

1. Nguyen, T. et al. (2019). *Forecasting Stock Prices Using HMM*. JRFM 12(4), 168. (팀원 참고 논문, `jrfm-12-00168-v2.pdf`)
2. Fischer, T. & Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions*. European Journal of Operational Research, 270(2), 654–669.
3. Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies, 33(5), 2223–2273.
4. Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting, 37(4), 1748–1764.

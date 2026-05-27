# 시장 국면 인식 기반 동적 포트폴리오 전략

> 금융 딥러닝 기초 기말 프로젝트 — 아주대학교 금융공학과

SPY 시계열 데이터에서 시장 국면(Bull / Neutral / Bear)을 학습하고, 이를 기반으로 포트폴리오 비중을 동적으로 조절하는 전략을 구성합니다. 정적 전략 및 규칙 기반 전략과 성과를 비교합니다.

---

## 전체 파이프라인

```
SPY + QQQ + GLD + TLT OHLCV (raw)
      │
      ▼
[1] HMM 라벨링          → SPY 기준 Bear / Neutral / Bull 국면 라벨
      │
      ▼
[2] 교차 자산 데이터셋 생성 → X: (698, 30, 40)  y: (698,)
      │                       (4자산 피처 concat, SPY 라벨만 사용)
      ▼
[3] Conv1D + LSTM 학습   → 시장 국면 분류 모델 (Accuracy 61.9%)
      │                     AdamW + Neutral-boost + Balanced Accuracy 기준 저장
      ▼
[4] 포트폴리오 백테스트   → Calmar Ratio 1.35 (전략 중 1위)
```

---

## 프로젝트 구조

```
.
├── README.md
├── jrfm-12-00168-v2.pdf          # 참고 논문
│
├── data/
│   ├── raw/
│   │   ├── spy_daily.csv         # SPY OHLCV (2010-01-04 ~ 2026-05-15)
│   │   ├── qqq_daily.csv         # QQQ OHLCV
│   │   ├── gld_daily.csv         # GLD OHLCV
│   │   └── tlt_daily.csv         # TLT OHLCV
│   └── processed/
│       ├── spy_hmm_regime_labels.csv       # 월간 HMM 라벨 (176개)
│       ├── spy_hmm_regime_labels_5d.csv    # 5거래일 HMM 라벨 (699개)
│       ├── *_hmm_regime_labels_5d.csv      # 자산별 HMM 라벨
│       ├── multi_asset_hmm_regime_labels_5d.csv # asset 컬럼 포함 통합 라벨
│       ├── spy_supervised_30d_5d.npz       # SPY 단일 자산 학습 데이터셋 (698샘플, 30×10)
│       ├── spy_supervised_30d_5d_index.csv # 샘플별 날짜 인덱스
│       ├── spy_supervised_30d_5d_meta.json # 데이터셋 메타정보
│       ├── cross_asset_supervised_30d_5d.npz # 교차 자산 피처 (698샘플, 30×40, SPY 라벨)
│       ├── multi_asset_supervised_30d_5d.npz # 다자산 딥러닝 학습 데이터셋 (2792샘플)
│       ├── multi_asset_supervised_30d_5d_index.csv
│       └── multi_asset_supervised_30d_5d_meta.json
│
├── scripts/
│   ├── hmm_regime_labeling.py         # [1] 단일 자산 HMM 라벨 생성
│   ├── generate_multi_asset_hmm_labels.py # [1-확장] 다자산 HMM 라벨 생성
│   ├── prepare_supervised_dataset.py  # [2] 학습 데이터셋 생성
│   ├── prepare_multi_asset_supervised_dataset.py # [2-확장] 다자산 학습 데이터셋 생성
│   ├── train.py                       # [3] 모델 학습
│   ├── experiments.py                 # [3] 4개 실험 비교
│   ├── backtest.py                    # [4] 포트폴리오 백테스트
│   └── visualize.py                   # 발표용 그래프 생성
│
├── outputs/
│   ├── models/
│   │   ├── best_model.pt              # 최종 모델 (Exp3)
│   │   └── model_Exp*.pt              # 실험별 모델
│   ├── figures/
│   │   ├── fig1_experiment_comparison.png
│   │   ├── fig2_cumulative_return.png
│   │   ├── fig3_strategy_metrics.png
│   │   └── fig4_confusion_matrix.png
│   └── results/
│       ├── experiment_results.json
│       ├── backtest_results.json
│       └── train_history.json
│
└── docs/
    ├── OVERVIEW.md           # 프로젝트 전체 설명
    ├── FOR_TEAMMATES.md      # 팀원용 요약
    ├── TRAINING_RESULTS.md   # 학습 과정 분석
    ├── MODEL_IMPROVEMENT.md  # 모델 개선 실험 상세
    ├── BACKTEST_RESULTS.md   # 백테스트 결과 분석
    └── MULTI_ASSET_LABELS.md # 다자산 라벨 생성 설명
```

---

## 실행 방법

### 요구사항

```bash
pip install numpy torch
```

HMM 라벨링 스크립트는 `numpy`만 사용합니다 (`pandas`, `sklearn`, `hmmlearn` 불필요).

### [1] HMM 라벨 생성 (팀원 담당)

```bash
python3 scripts/hmm_regime_labeling.py \
  --input data/raw/spy_daily.csv \
  --output data/processed/spy_hmm_regime_labels_5d.csv \
  --train-window 504 --fit-step 5 --states 3 \
  --smoothing-window 5 --target-horizon 1
```

### [1-확장] 다자산 HMM 라벨 생성

```bash
python3 scripts/generate_multi_asset_hmm_labels.py \
  --assets SPY QQQ GLD TLT \
  --start 2010-01-01 --end 2026-05-15
```

출력:

- `data/processed/{asset}_hmm_regime_labels_5d.csv`: 자산별 라벨
- `data/processed/multi_asset_hmm_regime_labels_5d.csv`: `asset` 컬럼을 붙인 통합 라벨

주의: 현재 `prepare_supervised_dataset.py`는 단일 `--raw`와 단일 `--labels`를 입력으로 받는다. 통합 라벨 CSV를 바로 넣으면 자산별 raw 데이터와 날짜가 섞이므로, 학습 데이터셋 확장 시에는 자산별 샘플을 만든 뒤 합치는 방식이 필요하다.

### [2] 지도학습 데이터셋 생성 (팀원 담당)

```bash
python3 scripts/prepare_supervised_dataset.py \
  --raw data/raw/spy_daily.csv \
  --labels data/processed/spy_hmm_regime_labels_5d.csv \
  --output data/processed/spy_supervised_30d_5d.npz \
  --index-output data/processed/spy_supervised_30d_5d_index.csv \
  --meta-output data/processed/spy_supervised_30d_5d_meta.json \
  --input-window 30 --target-horizon 1
```

### [2-확장] 다자산 지도학습 데이터셋 생성

```bash
python3 scripts/prepare_multi_asset_supervised_dataset.py
```

출력:

- `data/processed/multi_asset_supervised_30d_5d.npz`
- `data/processed/multi_asset_supervised_30d_5d_index.csv`
- `data/processed/multi_asset_supervised_30d_5d_meta.json`

이 스크립트는 자산별 `(raw, labels)` 쌍을 따로 샘플링한 뒤 합친다. split은 target date 기준 시간순으로 나누므로 같은 날짜의 SPY/QQQ/GLD/TLT 샘플이 서로 다른 split에 들어가지 않는다.

### [3] 모델 학습

```bash
python3 scripts/train.py          # 기본 학습 (best_model.pt 저장)
python3 scripts/experiments.py    # 4개 실험 전체 비교
```

#### `train.py` 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--data` | `data/processed/spy_supervised_30d_5d.npz` | 학습할 데이터 파일 경로 |
| `--model-output` | `outputs/models/best_model.pt` | 학습된 모델 저장 위치 |
| `--history-output` | `outputs/results/train_history.json` | 학습 기록(loss/acc) 저장 위치 |
| `--epochs` | `200` | 최대 학습 반복 횟수 |
| `--batch-size` | `16` | 한 번에 볼 샘플 수 |
| `--lr` | `3e-4` | 학습률 (learning rate) |
| `--patience` | `25` | 성능이 개선되지 않아도 기다릴 epoch 수 (early stopping) |

#### 예시

```bash
# SPY 단일 자산 기본 학습
python3 scripts/train.py

# 교차 자산(SPY+QQQ+GLD+TLT) 피처로 학습
python3 scripts/train.py \
  --data data/processed/cross_asset_supervised_30d_5d.npz \
  --model-output outputs/models/best_model_cross_asset.pt \
  --history-output outputs/results/train_history_cross_asset.json

# 다자산 라벨 데이터로 학습
python3 scripts/train.py \
  --data data/processed/multi_asset_supervised_30d_5d.npz \
  --model-output outputs/models/best_model_multi_asset.pt \
  --history-output outputs/results/train_history_multi_asset.json

# 하이퍼파라미터 직접 지정
python3 scripts/train.py \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-4 \
  --patience 50
```

### [4] 백테스트 및 시각화

```bash
python3 scripts/backtest.py       # 전략별 성과 비교
python3 scripts/visualize.py      # 발표용 그래프 생성
```

---

## 모델 구조: Conv1D + LSTM

```
입력 (batch, 30, 40)   ← SPY+QQQ+GLD+TLT 각 10개 피처 concat
      ↓
Conv1D × 2   — 3일 단위 국소 패턴 추출, 노이즈 완화
      ↓
LSTM         — 시간 흐름 학습, 장기 의존성 포착
      ↓
Linear + Softmax
      ↓
출력 (batch, 3)  →  [p_bear, p_neutral, p_bull]
```

**학습 전략**:
- AdamW optimizer (weight_decay=1e-2)
- Neutral class boost (×1.2) — 중간 국면 인식 강화
- Balanced Accuracy 기준 best model 저장
- Early stopping (patience=10)

포트폴리오 비중 결정:
```
w_stock = p_bull + 0.5 × p_neutral
w_cash  = 1 - w_stock
```

---

## 모델 개선 실험 결과

총 3단계(Phase) 실험을 진행하였습니다. 상세 내용은 [docs/MODEL_IMPROVEMENT.md](docs/MODEL_IMPROVEMENT.md)를 참고하세요.

| 단계 | 실험 | Accuracy | Bear | Neutral | Bull |
|------|------|---------|------|---------|------|
| Phase 1 | Baseline (SPY, 10 피처) | 57.1% | 34.9% | 23.8% | 97.6% |
| Phase 1 | Focal Loss + Augmentation | 61.0% | 46.5% | 33.3% | 90.2% |
| Phase 2 | 4자산 각자 라벨 (10 피처) | 59.8% | 58.8% | 25.3% | 80.6% |
| **Phase 3** | **Cross-asset 피처 + AdamW + Neutral-boost (최종)** | **61.9%** | **60.5%** | 0.0% | **95.1%** |

**최종 모델**: Cross-asset 피처(30×40) + AdamW + Neutral-boost 1.2, seed=42 고정, Accuracy **61.9%**

---

## 백테스트 결과 (2024.04 ~ 2026.05)

| 전략 | 누적수익 | Sharpe | MDD | Calmar |
|------|--------|--------|-----|--------|
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| MA Crossover | 29.1% | 0.82 | -10.9% | 1.19 |
| **Conv1D+LSTM (ours)** | **21.9%** | 0.73 | **-7.4%** | **1.35** |

테스트 기간이 강한 상승장(AI 붐, 미국 금리 인하)이어서 절대 수익률은 낮지만, **MDD -7.4%와 Calmar Ratio 1.35로 하락 방어 측면에서 모든 전략 중 1위**입니다.

---

## HMM 라벨링 상세 (팀원 파트)

### 방법

- 504거래일(2년) rolling window로 3-state Gaussian HMM 학습
- 각 state의 Sharpe ratio 비교 → Bull / Neutral / Bear 매핑
- 라벨 인코딩: `Bear=0`, `Neutral=1`, `Bull=2`

### 논문과의 차이

참고 논문(`jrfm-12-00168-v2.pdf`)의 핵심 아이디어(2년 rolling HMM + Sharpe 기반 국면 판단)를 따르되, 다음 차이가 있습니다.

- 논문은 월간 리밸런싱 / 현재 구현은 5거래일 간격
- 논문은 수익률만 사용 / 현재 구현은 변동성, MA 괴리, drawdown 추가
- Bear/Neutral/Bull은 절대 기준이 아닌 **rolling window 내 상대적 Sharpe ranking**

### 주요 컬럼 (`spy_hmm_regime_labels_5d.csv`)

| 컬럼 | 설명 |
|------|------|
| `hmm_label` | Bear / Neutral / Bull |
| `hmm_label_code` | 0 / 1 / 2 |
| `prob_bear/neutral/bull` | 현재 시점 state posterior 확률 |
| `state_sharpe` | 현재 state의 rolling window 내 Sharpe |
| `target_label_plus_1_steps` | 다음 라벨 (모델 예측 목표) |

---

## 참고 문헌

1. Nguyen, T. et al. (2019). *Forecasting Stock Prices Using HMM*. JRFM 12(4), 168.
2. Fischer, T. & Krauss, C. (2018). *Deep learning with LSTM networks for financial market predictions*. EJOR, 270(2), 654–669.
3. Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. RFS, 33(5), 2223–2273.
4. Lin, T. Y. et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.

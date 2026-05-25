# 시장 국면 인식 기반 동적 포트폴리오 전략

> 금융 딥러닝 기초 기말 프로젝트 — 아주대학교 금융공학과

SPY 시계열 데이터에서 시장 국면(Bull / Neutral / Bear)을 학습하고, 이를 기반으로 포트폴리오 비중을 동적으로 조절하는 전략을 구성합니다. 정적 전략 및 규칙 기반 전략과 성과를 비교합니다.

---

## 전체 파이프라인

```
SPY OHLCV (raw)
      │
      ▼
[1] HMM 라벨링          → Bear / Neutral / Bull 국면 라벨
      │
      ▼
[2] 지도학습 데이터셋 생성  → X: (698, 30, 10)  y: (698,)
      │
      ▼
[3] Conv1D + LSTM 학습   → 시장 국면 분류 모델 (Accuracy 61%)
      │
      ▼
[4] 포트폴리오 백테스트   → Calmar Ratio 1.30 (전략 중 1위)
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
│   │   └── spy_daily.csv         # SPY OHLCV (2010-01-04 ~ 2026-05-15)
│   └── processed/
│       ├── spy_hmm_regime_labels.csv       # 월간 HMM 라벨 (176개)
│       ├── spy_hmm_regime_labels_5d.csv    # 5거래일 HMM 라벨 (699개)
│       ├── spy_supervised_30d_5d.npz       # 딥러닝 학습 데이터셋
│       ├── spy_supervised_30d_5d_index.csv # 샘플별 날짜 인덱스
│       └── spy_supervised_30d_5d_meta.json # 데이터셋 메타정보
│
├── scripts/
│   ├── hmm_regime_labeling.py         # [1] HMM 라벨 생성
│   ├── prepare_supervised_dataset.py  # [2] 학습 데이터셋 생성
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
    └── BACKTEST_RESULTS.md   # 백테스트 결과 분석
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

### [3] 모델 학습

```bash
python3 scripts/train.py          # 기본 학습 (best_model.pt 저장)
python3 scripts/experiments.py    # 4개 실험 전체 비교
```

### [4] 백테스트 및 시각화

```bash
python3 scripts/backtest.py       # 전략별 성과 비교
python3 scripts/visualize.py      # 발표용 그래프 생성
```

---

## 모델 구조: Conv1D + LSTM

```
입력 (batch, 30, 10)
      ↓
Conv1D × 2   — 3일 단위 국소 패턴 추출, 노이즈 완화
      ↓
LSTM         — 시간 흐름 학습, 장기 의존성 포착
      ↓
Linear + Softmax
      ↓
출력 (batch, 3)  →  [p_bear, p_neutral, p_bull]
```

포트폴리오 비중 결정:
```
w_stock = p_bull + 0.5 × p_neutral
w_cash  = 1 - w_stock
```

---

## 모델 개선 실험 결과

| 실험 | 변경 내용 | Accuracy | Bear | Neutral | Bull |
|------|---------|---------|------|---------|------|
| Exp1 | Baseline | 57.1% | 34.9% | 23.8% | 97.6% |
| Exp2 | Focal Loss | 52.4% | 34.9% | 47.6% | 73.2% |
| **Exp3** | **Focal Loss + Data Augmentation** | **61.0%** | **46.5%** | 33.3% | 90.2% |
| Exp4 | BiLSTM + Attention | 49.5% | 30.2% | 38.1% | 75.6% |

**Data Augmentation**(학습 데이터 488→976개)이 가장 효과적이었습니다.

---

## 백테스트 결과 (2024.04 ~ 2026.05)

| 전략 | 누적수익 | Sharpe | MDD | Calmar |
|------|--------|--------|-----|--------|
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| MA Crossover | 29.1% | 0.82 | -10.9% | 1.19 |
| **Conv1D+LSTM (ours)** | **14.8%** | 0.33 | **-5.2%** | **1.30** |

테스트 기간이 강한 상승장(AI 붐, 미국 금리 인하)이어서 절대 수익률은 낮지만, **MDD -5.2%와 Calmar Ratio 1.30으로 하락 방어 측면에서 모든 전략 중 1위**입니다.

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

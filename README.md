# 시장 국면 인식 기반 동적 포트폴리오 전략

> 금융 딥러닝 기초 기말 프로젝트 — 아주대학교 금융공학과

Conv1D+LSTM으로 시장 국면(Bear/Neutral/Bull)을 분류하고, 국면 확률을 MVO(평균-분산 최적화)에 연결해 동적으로 자산 비중을 조절하는 전략을 구성합니다.

---

## 전체 파이프라인

```
SPY + QQQ + GLD + TLT OHLCV (raw)
      │
      ▼
[1] HMM 라벨링          → SPY 기준 Bear / Neutral / Bull 국면 라벨
      │                   (504일 rolling window, 3-state Gaussian HMM)
      ▼
[2] 교차 자산 데이터셋   → X: (698, 30, 40)  y: (698,)
      │                   (4자산 피처 concat, 5일 단위 리밸런싱)
      ▼
[3] Conv1D + LSTM 학습  → 시장 국면 분류 (Accuracy 61.9%, Bear Recall 60.5%)
      │                   AdamW + Neutral-boost 1.2 + seed=42
      ▼
[4] Regime-MVO 백테스트 → 훈련셋 국면별 Sharpe 최대화 비중 → 소프트 배분
                          MDD -7.2% (전략 중 최저), Calmar 2.16
```

---

## 핵심 컨셉

**"국면 분류기가 설명하는 포트폴리오 리밸런싱"**

- 단순 수익률 예측이 아닌 **시장 국면(Bear/Neutral/Bull) 분류** → 해석 가능
- 분류 확률을 MVO에 연결: `w = p_bear × w_bear + p_neutral × w_neutral + p_bull × w_bull`
- 국면별 최적 비중은 훈련 데이터에서 **Sharpe 최대화로 자동 계산**

```
Bear  → TLT 100%          (하락장: 채권으로 방어)
Neutral → SPY 51% + GLD 49%  (중립: 균형)
Bull  → SPY 95% + QQQ 5%    (상승장: 주식 집중)
```

---

## 모델 구조: Conv1D + LSTM

```
입력 (batch, 30, 40)   ← SPY+QQQ+GLD+TLT 각 10개 피처 concat
      ↓
Conv1D × 2   — 단기 국소 패턴 추출
      ↓
LSTM         — 시계열 장기 의존성 학습
      ↓
Linear + Softmax
      ↓
출력 (batch, 3)  →  [p_bear, p_neutral, p_bull]
```

| 하이퍼파라미터 | 값 |
|---|---|
| conv_channels | 16 |
| lstm_hidden | 32 |
| dropout | 0.6 |
| optimizer | AdamW (lr=1e-4, weight_decay=1e-2) |
| neutral_boost | 1.2 |
| best_metric | val_balanced_accuracy |
| seed | 42 |

---

## 실험 결과: 분류 성능

| 단계 | 실험 | Accuracy | Bear Recall | Neutral Recall | Bull Recall |
|---|---|---|---|---|---|
| Phase 1 | Baseline (SPY, 10피처) | 57.1% | 34.9% | 23.8% | 97.6% |
| Phase 1 | Augmentation | 61.0% | 46.5% | 33.3% | 90.2% |
| Phase 2 | 4자산 각자 라벨 | 59.8% | 58.8% | 25.3% | 80.6% |
| **Phase 3** | **Cross-asset + AdamW (최종)** | **61.9%** | **60.5%** | 0.0% | **95.1%** |

> Bear Recall 34% → 60%로 개선이 핵심. Neutral Recall 0%는 레이블 불명확성에 기인한 구조적 한계.

---

## 백테스트 결과 (Test: 2024.04 ~ 2026.05)

| 전략 | 누적수익 | Sharpe | MDD | Calmar |
|---|---|---|---|---|
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N (논문 벤치마크) | 50.9% | 1.41 | -8.8% | 2.47 |
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| MA Crossover | 29.1% | 0.82 | -10.9% | 1.19 |
| DL Regime SPY/Cash | 21.9% | 0.73 | -7.4% | 1.35 |
| Regime Momentum Tilt | 43.1% | 1.08 | -12.9% | 1.46 |
| **Regime-MVO (최종)** | **35.3%** | **1.10** | **-7.2%** | **2.16** |

> Regime-MVO: MDD -7.2%로 전략 중 최저. 수익은 EW보다 낮지만 하락 방어 + 리스크 조정 수익 균형.

### 하락장 검증 (2022 Bear Market, SPY -18.6%)

| 전략 | MDD |
|---|---|
| Buy & Hold | -20.5% |
| EW 1/N | -21.7% |
| MA Crossover | -20.8% |
| 60/40 | -12.5% |
| **DL Regime SPY/Cash** | **-10.5%** |
| Regime-MVO (최종) | -22.2% |

> 2022년(주식·채권 동반 하락)에서는 현금 비중을 둘 수 있는 SPY/Cash 전략이 낙폭 49%를 줄였다. 반면 Regime-MVO는 Bear 비중이 TLT에 집중되어 금리인상형 Bear에 취약했다.

---

## 출력 그림

| 그림 | 내용 |
|---|---|
| [fig01](outputs/figures/final/fig01_pipeline.png) | HMM → Conv1D+LSTM → Regime-MVO 파이프라인 |
| [fig02](outputs/figures/final/fig02_related_work.png) | 관련 연구 비교 |
| [fig03](outputs/figures/final/fig03_main_result.png) | 핵심 결과: MDD / Calmar 비교 |
| [fig04](outputs/figures/final/fig04_classification_performance.png) | Phase별 분류 성능 비교 |
| [fig05](outputs/figures/final/fig05_confusion_matrix.png) | Confusion Matrix (Phase 3 최종) |
| [fig06](outputs/figures/final/fig06_regime_conditional.png) | Bear / Neutral / Bull 구간별 전략 성과 |
| [fig07](outputs/figures/final/fig07_ablation.png) | Ablation: 구성요소별 기여 |
| [fig08](outputs/figures/final/fig08_2022_bear.png) | 2022 하락장 검증 |

---

## 프로젝트 구조

```
.
├── README.md
├── jrfm-12-00168-v2.pdf          # 참고 논문 (Kim et al. 2019)
├── RegimFolio.pdf                # 참고 논문 (Zhang et al. 2025)
│
├── data/
│   ├── raw/                      # SPY/QQQ/GLD/TLT OHLCV
│   └── processed/
│       ├── spy_hmm_regime_labels_5d.csv
│       ├── cross_asset_supervised_30d_5d.npz      # 최종 데이터셋 (698샘플, 30×40)
│       └── cross_asset_supervised_30d_5d_index.csv
│
├── scripts/
│   ├── hmm_regime_labeling.py              # HMM 라벨 생성
│   ├── prepare_cross_asset_dataset.py      # 교차 자산 데이터셋 생성
│   ├── train.py                            # 모델 학습
│   ├── backtest.py                         # 기본 백테스트
│   ├── backtest_regime_advanced.py         # Regime Momentum Tilt 백테스트
│   ├── backtest_mvo.py                     # Regime-MVO 백테스트 (최신)
│   ├── backtest_2022.py                    # 2022 하락장 백테스트
│   ├── regime_portfolio_policy.py          # Regime Momentum Tilt 정책
│   ├── visualize.py                        # fig1~4 생성
│   ├── visualize_comparison.py             # fig6 생성
│   └── experiments.py                      # 실험 비교
│
├── outputs/
│   ├── models/best_model.pt                # 최종 모델 (Phase 3, seed=42)
│   ├── figures/fig1~fig7.png
│   └── results/
│       ├── backtest_results.json
│       ├── backtest_regime_momentum_results.json
│       ├── backtest_mvo_results.json
│       └── backtest_2022_results.json
│
└── docs/
    ├── OVERVIEW.md               # 프로젝트 전체 설명
    ├── FOR_TEAMMATES.md          # 팀원용 요약
    ├── MODEL_IMPROVEMENT.md      # 실험 상세 기록
    └── PRESENTATION_OUTLINE.md  # 발표 목차 & Q&A 대응
```

---

## 실행 방법

```bash
# 최종 모델 재현
python3 scripts/train.py \
  --data data/processed/cross_asset_supervised_30d_5d.npz \
  --model-output outputs/models/best_model.pt \
  --epochs 80 --patience 10 --batch-size 16 \
  --lr 1e-4 --conv-channels 16 --lstm-hidden 32 \
  --dropout 0.6 --weight-decay 1e-2 \
  --neutral-boost 1.2 --best-metric val_bal_acc --seed 42

# 백테스트 (전략 비교)
python3 scripts/backtest_mvo.py              # Regime-MVO (최신)
python3 scripts/backtest_regime_advanced.py  # Regime Momentum Tilt
python3 scripts/backtest_2022.py             # 2022 하락장 검증

# 그래프 생성
python3 scripts/visualize.py
python3 scripts/visualize_comparison.py
```

---

## 관련 연구

| 논문 | 기여 | 우리와의 관계 |
|---|---|---|
| Kim et al. (2019) | HMM 기반 국면 레이블링 | 레이블링 방법론 계승 |
| Jiang et al. (2017) | DRL 포트폴리오 최적화 (391회 인용) | end-to-end 방향의 선행 연구 |
| Zhang et al. (2025) RegimeFolio | VIX 국면 + MVO + 섹터 앙상블 | 우리와 동일 구조, 국면 감지는 우리가 더 정교 |

---

## 한계 및 향후 연구

- **Neutral Recall 0%**: 중립 구간 레이블 불명확 → 라벨 정의 및 표본 확장 개선 필요
- **698샘플**: 금융 시계열 구조적 한계 (14년 × 5일 단위) → 일별 sliding으로 ~3,500샘플 확장 가능
- **파라미터 최적화**: MVO 계수를 Sharpe 직접 최대화로 도출 (완료) → end-to-end 학습으로 확장 가능
- **2022 환경**: 금리인상 Bear에서 TLT도 폭락 → 방어 자산을 단기채/MMF로 교체 필요

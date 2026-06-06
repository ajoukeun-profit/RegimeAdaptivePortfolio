# 시장 국면 인식 기반 동적 포트폴리오 전략

> 금융 딥러닝 기초 기말 프로젝트 — 아주대학교 금융공학과

HMM이 만든 시장 국면 pseudo-label을 Conv1D+LSTM으로 예측하고, 그 예측 확률을 MVO(Mean-Variance Optimization) 포트폴리오 비중에 연결해 동적으로 자산배분하는 프로젝트입니다.

현재 최종 전략은 **Binary Soft Label + 2-Regime MVO + weight cap 40%**입니다.

---

## 핵심 결론

처음에는 Bear / Neutral / Bull 3-class 분류로 시작했지만, Neutral label이 구조적으로 애매해 test에서 전혀 예측되지 않는 문제가 있었습니다. 그래서 투자 목적에 더 맞게 문제를 **Bear vs Non-Bear** binary classification으로 재정의했습니다.

최종 흐름은 다음과 같습니다.

```text
Neutral label failure
-> Bear vs Non-Bear binary classification
-> Binary soft-label training
-> 2-Regime MVO
-> MVO weight cap 40%
```

최종 비교 결과:

| Strategy | CumRet | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| **Binary Regime-MVO Soft cap 40%** | **53.7%** | **1.48** | **-9.0%** | **2.55** |

Binary soft-label MVO cap 40%가 현재 비교 전략 중 누적수익률, Sharpe, Calmar 기준으로 가장 균형 잡힌 결과를 냈습니다.

---

## 전체 파이프라인

```text
SPY + QQQ + GLD + TLT OHLCV
      |
      v
HMM 3-state pseudo-label
      |
      v
Bear / Neutral / Bull posterior
      |
      v
Binary soft label
P(Bear), P(Non-Bear) = P(Neutral) + P(Bull)
      |
      v
Conv1D + LSTM classifier
      |
      v
2-Regime MVO allocation
w_t = P(Non-Bear) * w_non_bear + P(Bear) * w_bear
      |
      v
Weight cap 40% portfolio backtest
```

---

## 왜 Binary로 바꿨나

3-class 모델의 문제:

| Metric | 3-class hard label |
|---|---:|
| Balanced Accuracy | 51.9% |
| Bear Recall | 60.5% |
| Neutral Recall | 0.0% |
| Bull Recall | 95.1% |

Neutral은 HMM의 중간 상태라 Bear/Bull보다 경제적 경계가 흐립니다. 이 프로젝트의 최종 목표도 모든 국면을 예쁘게 맞히는 것이 아니라 하방 위험을 관리하는 것이므로, Neutral과 Bull을 합쳐 **Non-Bear**로 두는 편이 더 자연스럽습니다.

Binary 실험 결과:

| Model | Test Balanced Acc | Bear Recall |
|---|---:|---:|
| 3-class hard label | 51.9% | 60.5% |
| Binary hard label | 70.2% | 58.1% |
| **Binary soft label** | **72.4%** | **67.4%** |

Binary soft label은 hard label보다 Bear 탐지가 좋아졌고, 이 확률을 최종 2-Regime MVO에 연결했습니다.

---

## 딥러닝이 필요한가

작은 샘플에서는 Logistic Regression이나 Random Forest 같은 classical baseline도 꼭 확인해야 합니다. 그래서 LR/RF와 Conv1D+LSTM을 같은 binary Bear detection task에서 비교했습니다.

| Model | Valid Balanced Acc | Test Balanced Acc | Bear Recall |
|---|---:|---:|---:|
| Logistic Regression | 63.7% | 61.4% | 32.6% |
| Random Forest | 79.1% | 66.3% | 53.5% |
| Conv1D+LSTM Binary | 85.2% | 70.2% | 58.1% |
| **Conv1D+LSTM Binary Soft** | **73.3%** | **72.4%** | **67.4%** |

LR < RF < LSTM 흐름이 확인되어, 시계열 패턴을 학습하는 딥러닝 모델의 효과가 있다고 볼 수 있습니다.

---

## MVO Weight Cap

제약 없는 MVO는 작은 샘플에서 추정 오차에 민감하게 반응해 특정 자산에 몰빵하는 문제가 있습니다. Binary MVO에서도 cap이 없으면 다음처럼 극단적 비중이 나옵니다.

| Cap | Non-Bear MVO | Bear MVO |
|---|---|---|
| None | SPY 100% | TLT 100% |
| 50% | SPY 50%, QQQ 50% | GLD 50%, TLT 50% |
| 40% | SPY 40%, QQQ 40%, GLD 20% | GLD 40%, TLT 40%, SPY 20% |

따라서 최종 전략에서는 자산별 최대 비중을 40%로 제한했습니다.

---

## 출력 그림

발표/보고서에는 `outputs/figures/final/` 아래 그림만 사용합니다.

| Figure | File | 내용 |
|---|---|---|
| Fig 01 | [fig01_pipeline.png](outputs/figures/final/fig01_pipeline.png) | HMM → Conv1D+LSTM → Regime-MVO 파이프라인 |
| Fig 02 | [fig02_related_work.png](outputs/figures/final/fig02_related_work.png) | 관련 연구 비교 |
| Fig 03-A | [fig03_static_dynamic_backtest.png](outputs/figures/final/fig03_static_dynamic_backtest.png) | Binary Regime-MVO Soft cap 40% 누적수익률 / drawdown |
| Fig 03-B | [fig03_main_result.png](outputs/figures/final/fig03_main_result.png) | 최종 전략 핵심 성과 요약 |
| Fig 04 | [fig04_classification_performance.png](outputs/figures/final/fig04_classification_performance.png) | Binary Hard / Binary Soft / LR / RF 비교 |
| Fig 05 | [fig05_confusion_matrix.png](outputs/figures/final/fig05_confusion_matrix.png) | Binary Soft Label confusion matrix |
| Fig 07 | [fig07_ablation.png](outputs/figures/final/fig07_ablation.png) | 전략 ablation 비교 |
| Fig 09 | [fig09_binary_mvo_weights.png](outputs/figures/final/fig09_binary_mvo_weights.png) | MVO cap별 비중 변화 |

---

## 프로젝트 구조

```text
.
├── data/
│   ├── raw/
│   └── processed/
│       ├── spy_hmm_regime_labels_5d.csv
│       ├── cross_asset_supervised_30d_5d.*
│       ├── cross_asset_supervised_30d_5d_binary_bear.*
│       └── cross_asset_supervised_30d_5d_binary_soft_labels.*
├── docs/
│   ├── PRESENTATION_OUTLINE.md
│   ├── TROUBLESHOOTING.md
│   ├── VISUALIZATION_PLAN.md
│   └── CONCEPTUAL_MATH_REPORT_NOTION.md
├── outputs/
│   ├── figures/final/
│   ├── models/
│   └── results/
└── scripts/
    ├── hmm_regime_labeling.py
    ├── prepare_cross_asset_dataset.py
    ├── prepare_supervised_dataset.py
    ├── train.py
    ├── train_soft_labels.py
    ├── baseline_binary_bear_sklearn.py
    ├── backtest_mvo.py
    ├── backtest_binary_mvo.py
    ├── regime_portfolio_policy.py
    ├── visualize_binary_mvo_results.py
    ├── visualize_pipeline.py
    └── visualize_related_work.py
```

`FOR_TEAMMATES.md`는 삭제했습니다. 팀원 공유용 요약은 현재 `README.md`, `docs/PRESENTATION_OUTLINE.md`, `docs/TROUBLESHOOTING.md`로 통일합니다.

---

## 주요 실행 명령

```bash
# 1. Binary soft-label dataset 생성
python3 scripts/prepare_cross_asset_dataset.py \
  --binary-bear --soft-labels \
  --output data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz \
  --index-output data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_index.csv \
  --meta-output data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_meta.json

# 2. Binary soft-label Conv1D+LSTM 학습
python3 scripts/train_soft_labels.py \
  --data data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz \
  --index data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_index.csv \
  --model-output outputs/models/best_model_binary_soft_labels.pt \
  --history-output outputs/results/train_history_binary_soft_labels.json

# 3. 최종 Binary Soft MVO cap 40% 백테스트
python3 scripts/backtest_binary_mvo.py \
  --max-weight 0.4 \
  --output outputs/results/backtest_binary_soft_mvo_cap40_results.json

# 4. 최종 figure 생성
python3 scripts/visualize_binary_mvo_results.py
python3 scripts/visualize_pipeline.py
python3 scripts/visualize_related_work.py
```

---

## 한계 및 다음 작업

- HMM label은 실제 정답이 아니라 pseudo-label이므로 classification 성능을 과대해석하면 안 됩니다.
- cap 40%는 현재 test 구간에서 좋은 결과이며, 더 엄밀하게는 validation/walk-forward 방식으로 cap을 선택해야 합니다.
- MVO는 평균과 공분산 추정에 민감하므로 shrinkage, equal-weight blending, 현금/단기채 추가를 후속 실험으로 볼 수 있습니다.
- 최종 결론은 "항상 수익률을 극대화했다"가 아니라, **binary regime probability와 capped MVO를 결합했을 때 현재 비교군 중 가장 균형 잡힌 risk-return 결과를 얻었다**입니다.

# 15분 발표 개요: Market Regime-Aware Dynamic Portfolio Strategy

> 한국어 제목: 시장 국면 인식 기반 동적 포트폴리오 전략

## 0. 발표 핵심 메시지

본 프로젝트는 HMM이 만든 시장 국면 pseudo-label을 Conv1D+LSTM으로 예측하고, 그 예측 확률을 MVO 포트폴리오 비중에 연결하는 전략을 검증한다.

Troubleshooting 이후 최종 메시지는 다음과 같다.

```text
Neutral label failure
-> Bear vs Non-Bear binary classification
-> Binary soft-label training
-> 2-Regime MVO
-> MVO weight cap 40%
```

최종 후보 전략:

```text
Binary Regime-MVO Soft, cap 40%
```

핵심 결과:

| Strategy | CumRet | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

발표에서 강조할 점:

- HMM label은 실제 정답이 아니라 pseudo-label이다.
- 분류 정확도 자체보다 portfolio downstream 성과가 중요하다.
- Neutral은 구조적으로 애매해서 binary Bear detection으로 재정의했다.
- Soft label은 Bear 탐지를 개선했다.
- MVO cap은 작은 샘플에서 발생하는 자산 몰빵을 완화했다.
- 단일 test 구간 결과이므로 과도한 일반화는 피한다.

## 1. 15분 발표 구조

| Section | 내용 | 시간 | Slide |
|---|---|---:|---:|
| 1. Background | 왜 regime-aware portfolio가 필요한가 | 2분 | 1-2 |
| 2. Methodology | HMM pseudo-label, Conv1D+LSTM, Regime-MVO | 3분 | 3-5 |
| 3. Problem Diagnosis | Neutral failure, pseudo-label, MVO estimation error | 3분 | 6-7 |
| 4. Experiments | Binary, baseline, soft label, MVO cap | 3분 | 8-9 |
| 5. Final Results | Binary Soft 2-Regime MVO cap 40% | 3분 | 10-11 |
| 6. Conclusion | 결론, 한계, 향후 개선 | 1분 | 12 |

## 2. Slide-by-Slide Outline

### Slide 1. Title

제목:

```text
Market Regime-Aware Dynamic Portfolio Strategy
```

말할 내용:

- 가격을 직접 예측하는 대신 시장 국면을 예측한다.
- 예측된 국면 확률을 포트폴리오 비중 조절에 사용한다.

### Slide 2. Background: 왜 정적 포트폴리오가 부족한가

핵심 질문:

> 시장 상황이 계속 바뀌는데, 하나의 고정 포트폴리오가 충분한가?

포인트:

- 전통적 MVO는 과거 데이터에서 기대수익률과 공분산을 추정한다.
- 금융시장은 non-stationary하다.
- 상승장, 하락장, 중립장에서는 자산 간 관계가 달라질 수 있다.
- 따라서 regime-aware allocation이 필요하다.

### Slide 3. Pipeline

Figure:

- [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png)

파이프라인:

```text
SPY/QQQ/GLD/TLT OHLCV
-> HMM pseudo-label
-> 30 days x 40 features
-> Conv1D+LSTM
-> regime probabilities
-> Regime-MVO allocation
```

주의:

- HMM label은 ground truth가 아니라 pseudo-label이다.

### Slide 4. Original 3-Class Setup

기존 설정:

```text
Bear / Neutral / Bull
```

3-class 모델 결과:

| Metric | Value |
|---|---:|
| Accuracy | 61.9% |
| Balanced Accuracy | 51.9% |
| Bear Recall | 60.5% |
| Neutral Recall | 0.0% |
| Bull Recall | 95.1% |

해석:

- Bear/Bull은 어느 정도 학습된다.
- Neutral은 test에서 한 번도 예측되지 않는다.
- Neutral은 경제적으로도 경계가 애매한 중간 국면이다.

### Slide 5. Problem Diagnosis

문제 3개:

| Problem | Evidence | Fix Direction |
|---|---|---|
| Neutral label failure | Neutral Recall 0.0% | Bear vs Non-Bear |
| HMM pseudo-label uncertainty | HMM label is model-generated | Soft label |
| Small-sample MVO estimation error | MVO picks extreme weights | Weight cap |

Gradient 문제:

- `train.py`에서 gradient clipping 적용 중이다.
- loss divergence나 NaN 증거는 없다.
- 현재 핵심 문제는 gradient보다 label definition과 MVO estimation이다.

### Slide 6. Experiment 1: Binary Bear vs Non-Bear

목적:

Neutral ambiguity를 제거하고 downside-risk detection에 집중한다.

결과:

| Model | Balanced Accuracy | Bear Recall |
|---|---:|---:|
| 3-class hard label | 51.9% | 60.5% |
| Binary hard label | 70.2% | 58.1% |

해석:

- Balanced Accuracy가 51.9%에서 70.2%로 크게 개선됐다.
- Bear Recall은 약간 낮지만, binary task가 훨씬 안정적이다.

### Slide 7. Experiment 2: LR/RF Baseline

목적:

작은 샘플에서 딥러닝이 정말 필요한지 확인한다.

결과:

| Model | Test Balanced Accuracy | Bear Recall |
|---|---:|---:|
| Logistic Regression | 61.4% | 32.6% |
| Random Forest | 66.3% | 53.5% |
| Binary Conv1D+LSTM | 70.2% | 58.1% |

해석:

- LR < RF < LSTM 순서다.
- 시계열 패턴을 학습하는 Conv1D+LSTM의 효과가 있다.

### Slide 8. Experiment 3: Binary Soft Label

Figure:

- [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png)
- [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png)

Soft target:

```text
P(Non-Bear) = P(Neutral) + P(Bull)
P(Bear)     = P(Bear)
```

결과:

| Model | Balanced Accuracy | Bear Recall |
|---|---:|---:|
| Binary hard label | 70.2% | 58.1% |
| Binary soft label | 72.4% | 67.4% |

해석:

- Binary soft label은 Bear 탐지를 개선했다.
- Bear Recall이 58.1%에서 67.4%로 상승했다.
- 이 확률을 최종적으로 2-Regime MVO에 연결한다.

### Slide 9. Experiment 4: MVO Cap

Figure:

- [fig09_binary_mvo_weights.png](../outputs/figures/final/fig09_binary_mvo_weights.png)

문제:

제약 없는 MVO는 작은 샘플에서 극단적 비중을 선택한다.

Binary MVO weight example:

| Cap | Non-Bear MVO | Bear MVO |
|---:|---|---|
| 100% | SPY 100.0% | TLT 100.0% |
| 50% | SPY 50.0%, QQQ 49.8%, GLD 0.2% | QQQ 7.3%, GLD 42.7%, TLT 50.0% |
| 40% | SPY 40.0%, QQQ 40.0%, GLD 17.3%, TLT 2.7% | QQQ 20.0%, GLD 40.0%, TLT 40.0% |

해석:

- cap이 없으면 자산 몰빵이 발생한다.
- cap 40%는 비중을 분산시켜 추정 오차 민감도를 낮춘다.

### Slide 10. Final Strategy: Binary Soft Label + 2-Regime MVO

최종 전략:

```text
w_t =
P(Non-Bear) * w_NonBear
+ P(Bear) * w_Bear
```

즉:

```text
Binary Soft Label model
-> P(Bear), P(Non-Bear)
-> 2-Regime MVO
-> cap 40%
```

Figure:

- [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png)

### Slide 11. Final Results

Figure:

- [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png)
- [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png)

결과:

| Strategy | CumRet | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

해석:

- Binary Soft MVO cap 40%가 현재 비교 전략 중 누적수익률이 가장 높다.
- EW 1/N 대비 누적수익률은 +2.8%p 높고 MDD는 거의 비슷하다.
- 3-class capped MVO 대비 누적수익률과 Calmar가 소폭 개선된다.
- 단일 test 구간 결과이므로 일반화에는 주의한다.

### Slide 12. Conclusion

결론:

1. Neutral은 구조적으로 애매했고, binary Bear detection이 더 안정적이었다.
2. LR/RF baseline보다 Conv1D+LSTM이 Bear detection에서 더 나았다.
3. Binary soft label은 Bear Recall을 개선했다.
4. MVO cap은 자산 몰빵을 완화했다.
5. 최종적으로 Binary Soft Label + 2-Regime MVO + cap 40%가 가장 균형 잡힌 결과를 냈다.

한계:

- HMM label은 pseudo-label이다.
- sample 수가 작다.
- cap 40%는 현재 test 구간에서 좋은 결과이며, 더 엄밀하게는 walk-forward validation이 필요하다.
- 방어 자산에 cash/short-term bond를 추가하면 더 안정적일 수 있다.

마지막 문장:

> 이 프로젝트의 핵심은 가격을 정확히 맞히는 것이 아니라, 위험 국면을 인식하고 그 확률을 포트폴리오 비중 조절에 연결하는 것이다.

## 3. 사용할 최종 Figure

| Figure | File | 역할 |
|---|---|---|
| Fig 01 | [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png) | 전체 파이프라인 |
| Fig 02 | [fig02_related_work.png](../outputs/figures/final/fig02_related_work.png) | 선행연구 |
| Fig 03-A | [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png) | 최종 전략 누적수익률 / drawdown 경로 |
| Fig 03-B | [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png) | 최종 성과 요약 |
| Fig 04 | [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png) | 분류 성능 개선 |
| Fig 05 | [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png) | Binary soft label confusion matrix |
| Fig 07 | [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png) | Ablation / benchmark |
| Fig 09 | [fig09_binary_mvo_weights.png](../outputs/figures/final/fig09_binary_mvo_weights.png) | MVO cap 효과 |

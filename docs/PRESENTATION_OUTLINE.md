# 발표 목차: 시장 국면 인식 기반 동적 포트폴리오 전략

> 영문 부제: Market Regime-Aware Dynamic Portfolio Strategy

## 0. 발표 전체 메시지

본 프로젝트는 가격 자체를 직접 예측하는 대신, 시장이 위험 국면인지 아닌지를 확률적으로 예측하고, 그 예측 확률을 포트폴리오 비중 조절에 연결하는 동적 자산배분 전략을 검증한다.

최종 전략은 다음이다.

```text
Binary Soft Label + 2-Regime MVO + Weight Cap 40%
```

전체 논리 흐름:

```text
정적 포트폴리오의 한계
-> 시장 국면 인식 필요
-> HMM으로 Bear / Neutral / Bull pseudo-label 생성
-> Neutral label 실패 확인
-> Bear vs Non-Bear binary task로 재정의
-> HMM posterior를 binary soft label로 변환
-> Conv1D+LSTM으로 Bear 확률 예측
-> 예측 확률로 2-Regime MVO 비중 결합
-> Weight cap 40%로 MVO 몰빵 완화
-> 최종 backtest 성과 평가
```

발표의 핵심 주장:

1. 금융시장은 국면에 따라 자산의 위험과 관계가 달라지므로 고정 비중만으로는 한계가 있다.
2. HMM label은 실제 정답이 아니라 pseudo-label이므로, 분류 성능은 과대해석하면 안 된다.
3. 3-class의 Neutral은 경제적 경계가 애매했고, test에서 한 번도 예측되지 않았다.
4. 투자 목적상 모든 국면을 세밀하게 맞히는 것보다 Bear를 탐지하는 것이 더 중요하다.
5. Binary soft label은 HMM posterior의 불확실성을 보존하면서 Bear Recall을 개선했다.
6. 예측 확률을 2-Regime MVO에 연결하고, cap 40%를 적용했을 때 현재 비교군 중 가장 균형 잡힌 risk-return 결과를 얻었다.

최종 결과:

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

발표 마지막에 가져갈 결론 문장:

> 이 프로젝트의 핵심은 시장을 완벽히 예측하는 것이 아니라, 위험 국면의 확률을 인식하고 그 확률을 포트폴리오 비중 조절에 연결하는 것이다.

## 1. 장별 구성 요약

사용자가 설정한 큰 틀을 유지해 4개 장으로 구성한다. 각 장은 단순히 목차를 나열하는 것이 아니라, 앞 장의 질문이 다음 장의 방법론과 실험으로 자연스럽게 이어지도록 설계한다.

| 장 | 핵심 질문 | 세부 목차 | 권장 시간 | 슬라이드 |
|---|---|---|---:|---:|
| 1. 연구 배경, 선행연구 | 왜 국면 인식 포트폴리오가 필요한가? | 연구 질문, 정적 포트폴리오 한계, 선행연구, 본 연구 차별점 | 3분 | 1-4 |
| 2. 방법론 | 국면 확률을 어떻게 만들고 비중으로 바꾸는가? | 전체 파이프라인, HMM pseudo-label, binary 재정의, Conv1D+LSTM, soft label, 2-Regime MVO | 5분 | 5-10 |
| 3. 실험설계 | 어떤 기준으로 전략을 검증하는가? | 데이터, train/test, baseline, ablation, 평가 지표 | 3분 | 11-13 |
| 4. 결과 분석, 한계와 결론 | 실제로 좋아졌는가, 어디까지 믿을 수 있는가? | 분류 결과, MVO cap 효과, backtest 결과, 한계, 결론 | 4분 | 14-17 |

총 17장 기준이다. 15분 발표에서는 한 장당 평균 45-55초를 잡고, Slide 14-16 결과 파트에 시간을 조금 더 배분한다.

## 2. 장별 세부 목차

## I. 연구 배경, 선행연구

이 장의 역할은 "왜 이 프로젝트를 해야 하는가"를 설득하는 것이다. 단순히 HMM과 LSTM을 썼다는 기술 소개로 시작하지 말고, 먼저 정적 포트폴리오의 한계와 시장 국면 인식의 필요성을 제시한다.

장 전체 흐름:

```text
정적 자산배분의 한계
-> 시장 국면이 바뀌면 위험/상관관계도 바뀜
-> 기존 연구는 국면 추정, 딥러닝 예측, 포트폴리오 최적화를 다룸
-> 본 프로젝트는 이 셋을 하나의 투자 파이프라인으로 연결
```

### Slide 1. 제목 및 연구 질문

슬라이드 제목:

```text
Market Regime-Aware Dynamic Portfolio Strategy
```

한 장의 메시지:

> 시장 국면을 예측하고, 그 확률을 포트폴리오 비중 조절에 연결할 수 있는가?

세부 목차:

- 연구 주제: 시장 국면 인식 기반 동적 포트폴리오 전략
- 핵심 질문: "시장 상태가 바뀔 때 포트폴리오도 바뀌어야 하는가?"
- 접근 방식: 국면 예측 확률을 자산배분 비중에 연결
- 사용 자산: SPY, QQQ, GLD, TLT
- 최종 키워드: HMM, Conv1D+LSTM, Binary Soft Label, MVO

발표 포인트:

- 이 발표는 단순 분류 모델 발표가 아니라 투자 전략 검증 발표다.
- 모델의 예측 결과가 최종적으로 backtest 성과로 평가된다는 점을 처음부터 명확히 한다.

### Slide 2. 연구 배경: 정적 포트폴리오의 한계

한 장의 메시지:

> 금융시장은 non-stationary하므로 하나의 고정 비중이 모든 구간에서 최적이기 어렵다.

세부 목차:

- 정적 포트폴리오 예시
  - Buy & Hold
  - 60/40
  - EW 1/N
- 정적 포트폴리오의 가정
  - 과거 평균과 공분산이 미래에도 유지된다고 가정
  - 자산 간 상관관계가 크게 변하지 않는다고 가정
- 실제 시장의 문제
  - 상승장과 하락장에서 위험 구조가 달라짐
  - 변동성 확대 구간에서 손실 집중 가능
  - 주식, 채권, 금의 역할이 국면별로 달라질 수 있음

발표 포인트:

- 정적 포트폴리오가 나쁘다는 주장이 아니라, 시장 상태 변화에 대응하지 못한다는 한계를 지적한다.
- 이 한계가 "국면 인식"이라는 연구 동기로 이어진다.

### Slide 3. 연구 목적: 예측을 투자 의사결정으로 연결

한 장의 메시지:

> 이 프로젝트의 목표는 국면을 맞히는 것에서 끝나는 것이 아니라, 예측 확률을 실제 투자 비중으로 변환하는 것이다.

세부 목차:

- 일반적인 예측 문제
  - 가격 상승/하락 예측
  - class accuracy 중심 평가
- 본 프로젝트의 예측 문제
  - Bear probability 예측
  - 예측 확률을 포트폴리오 비중에 반영
- 최종 평가 관점
  - classification: Balanced Accuracy, Bear Recall
  - portfolio: cumulative return, Sharpe, MDD, Calmar

핵심 흐름:

```text
Regime Prediction
-> Regime Probability
-> Portfolio Weight
-> Backtest Performance
```

발표 포인트:

- 분류 정확도는 중간 지표이고 최종 목적은 포트폴리오 성과다.
- Bear Recall을 보는 이유는 하방 위험 관리가 투자 목적과 직접 연결되기 때문이다.

### Slide 4. 선행연구와 본 연구의 차별점

Figure:

- [fig02_related_work.png](../outputs/figures/final/fig02_related_work.png)

한 장의 메시지:

> 기존 연구의 세 흐름을 결합해, 국면 추정부터 포트폴리오 배분까지 이어지는 하나의 실험 파이프라인을 만든다.

세부 목차:

- 선행연구 흐름 1: 시장 국면 추정
  - HMM 등 잠재 상태 모델로 시장 상태를 추정
- 선행연구 흐름 2: 딥러닝 기반 금융 시계열 예측
  - CNN/LSTM 계열 모델로 시계열 패턴 학습
- 선행연구 흐름 3: 포트폴리오 최적화
  - MVO로 기대수익률과 위험을 반영한 비중 산출
- 본 연구의 차별점
  - HMM은 label generator로 사용
  - Conv1D+LSTM은 미래 Bear 확률 예측에 사용
  - MVO는 예측 확률을 투자 비중으로 변환하는 downstream module로 사용

비교 표:

| 흐름 | 기존 접근 | 본 프로젝트 |
|---|---|---|
| 국면 인식 | HMM으로 시장 상태 추정 | HMM label을 pseudo-label로 사용 |
| 예측 모델 | 가격/수익률 또는 국면 예측 | Conv1D+LSTM으로 미래 Bear 확률 예측 |
| 포트폴리오 | 정적 MVO 또는 규칙 기반 전환 | 확률 가중 2-Regime MVO |

발표 포인트:

- 본 연구는 완전히 새로운 알고리즘을 제안한다기보다, 기존 방법들을 투자 의사결정 흐름에 맞게 연결한다.
- 이 연결 구조가 이후 methodology 파트의 중심이다.

## II. 방법론

이 장의 역할은 "어떻게 국면 확률을 만들고, 그 확률을 포트폴리오 비중으로 바꾸는가"를 설명하는 것이다. 핵심은 HMM label이 pseudo-label이라는 점, 그리고 3-class 실패 이후 binary soft label로 재정의했다는 점이다.

장 전체 흐름:

```text
데이터 입력
-> HMM pseudo-label
-> 3-class 문제 진단
-> Binary Bear detection
-> Conv1D+LSTM 확률 예측
-> 2-Regime MVO 확률 가중 비중
-> Weight cap 안정화
```

### Slide 5. 전체 방법론 파이프라인

Figure:

- [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png)

한 장의 메시지:

> 본 프로젝트는 HMM, Conv1D+LSTM, MVO를 순차적으로 연결한 투자 전략 파이프라인이다.

세부 목차:

- 입력 데이터
  - SPY, QQQ, GLD, TLT OHLCV
  - 최근 30거래일 cross-asset feature
- label 생성
  - HMM으로 Bear / Neutral / Bull pseudo-label 생성
  - HMM posterior probability 저장
- 예측 모델
  - Conv1D+LSTM으로 미래 5거래일 후 Bear 확률 예측
- 포트폴리오 배분
  - Non-Bear MVO와 Bear MVO 비중 계산
  - 예측 확률로 두 비중을 가중 평균
- backtest
  - 최종 동적 비중으로 성과 평가

파이프라인:

```text
SPY/QQQ/GLD/TLT OHLCV
-> HMM pseudo-label
-> 30 days x 40 features
-> Conv1D+LSTM
-> P(Non-Bear), P(Bear)
-> 2-Regime MVO allocation
-> Backtest
```

발표 포인트:

- 각 단계가 독립적인 실험이 아니라 하나의 투자 전략으로 연결된다.
- 마지막 성과가 좋아야 앞 단계의 예측도 의미를 가진다.

### Slide 6. HMM Pseudo-label 생성

한 장의 메시지:

> HMM은 관측되지 않는 시장 상태를 추정하지만, 그 결과는 실제 정답이 아니라 모델이 만든 pseudo-label이다.

세부 목차:

- HMM의 역할
  - 관측 데이터에서 잠재 시장 상태 추정
  - 상태를 Bear / Neutral / Bull로 사후 해석
- pseudo-label의 의미
  - 사람이 직접 labeling한 정답이 아님
  - HMM의 통계적 추정 결과
  - classification 성능을 절대적 진실처럼 해석하면 안 됨
- posterior probability의 활용
  - hard label: 가장 확률이 높은 state만 사용
  - soft label: 각 state 확률을 학습 target으로 활용

수식:

```text
s_t in {Bear, Neutral, Bull}
f_theta(X_t) ~= y_hat_{t+h}^{HMM}
```

발표 포인트:

- HMM label을 "정답"이라고 부르지 않고 "pseudo-label"이라고 계속 표현한다.
- 이 점이 soft label로 넘어가는 논리적 근거다.

### Slide 7. 3-class 문제 진단: Neutral Label Failure

한 장의 메시지:

> 초기 3-class 설정은 Neutral을 전혀 예측하지 못했고, 이는 label 정의 자체의 문제로 해석된다.

세부 목차:

- 초기 문제 정의
  - Bear / Neutral / Bull 3-class classification
- 관찰된 문제
  - Neutral Recall 0.0%
  - test set에서 Neutral을 한 번도 예측하지 않음
- 원인 해석
  - Neutral은 Bear와 Bull 사이의 중간 상태
  - HMM state의 경제적 경계가 명확하지 않음
  - 모델이 Neutral을 Bear 또는 Bull로 흡수
- 결론
  - class weight 조정보다 문제 재정의가 필요

결과 표:

| 지표 | 값 |
|---|---:|
| Accuracy | 61.9% |
| Balanced Accuracy | 51.9% |
| Bear Recall | 60.5% |
| Neutral Recall | 0.0% |
| Bull Recall | 95.1% |

발표 포인트:

- 이 슬라이드는 발표의 전환점이다.
- "모델이 나쁘다"가 아니라 "Neutral이라는 target 자체가 투자 목적에 덜 맞다"로 설명한다.

### Slide 8. Binary Bear Detection으로 재정의

한 장의 메시지:

> 투자 목적에 맞게 문제를 Bear vs Non-Bear로 단순화하면, 하방 위험 탐지에 더 집중할 수 있다.

세부 목차:

- 기존 설정

```text
Bear / Neutral / Bull
```

- 변경 설정

```text
Bear / Non-Bear
Non-Bear = Neutral + Bull
```

- 재정의 이유
  - 포트폴리오에서는 하락 위험 탐지가 핵심
  - Neutral과 Bull의 세부 구분보다 Bear 여부가 더 중요
  - binary 확률은 2-Regime MVO와 자연스럽게 연결됨
- binary hard label 결과
  - Balanced Accuracy: 70.2%
  - Bear Recall: 58.1%

비교 표:

| 모델 | Balanced Accuracy | Bear Recall |
|---|---:|---:|
| 3-class hard label | 51.9% | 60.5% |
| Binary hard label | 70.2% | 58.1% |

발표 포인트:

- Bear Recall만 보면 3-class가 약간 높아 보이지만, 3-class는 Neutral을 전혀 예측하지 못한다.
- binary는 목적이 더 명확하고 portfolio layer와 잘 맞는다.

### Slide 9. Conv1D+LSTM 국면 예측 모델

한 장의 메시지:

> Conv1D+LSTM은 최근 30일의 cross-asset 시계열 패턴을 이용해 미래 Bear 확률을 예측한다.

세부 목차:

- 입력 구조
  - 30거래일 sequence
  - 4개 자산 기반 40개 feature
  - cross-asset relative movement 반영
- 모델 구조
  - Conv1D: 짧은 구간의 local pattern 추출
  - LSTM: 시간 순서와 누적 흐름 학습
  - classifier: P(Non-Bear), P(Bear) 출력
- 학습 안정화
  - dropout
  - early stopping
  - gradient clipping
  - AdamW
- 출력의 의미
  - class label만이 아니라 portfolio weight에 들어가는 probability

발표 포인트:

- 모델의 출력값은 "맞다/틀리다" 판단으로 끝나지 않고 MVO 비중 혼합에 사용된다.
- 그래서 확률의 품질이 downstream 성과와 연결된다.

### Slide 10. Binary Soft Label과 2-Regime MVO

한 장의 메시지:

> HMM posterior를 binary soft target으로 합치고, 예측 확률로 Non-Bear/Bear MVO 비중을 가중 평균한다.

세부 목차:

- soft label 생성
  - HMM posterior의 불확실성을 보존
  - Neutral과 Bull posterior를 Non-Bear로 합산
- soft target

```text
P(Bear) = P(Bear)
P(Non-Bear) = P(Neutral) + P(Bull)
```

- 2-Regime MVO
  - Non-Bear 구간의 MVO 비중 계산
  - Bear 구간의 MVO 비중 계산
  - 매 시점 예측 확률로 두 비중을 혼합
- 최종 비중

```text
w_t =
P(Non-Bear) * w_NonBear
+ P(Bear) * w_Bear
```

- 제약조건

```text
sum(w_i) = 1
0 <= w_i <= 0.4
```

발표 포인트:

- hard label은 애매한 샘플도 하나의 class로 강제하지만, soft label은 불확실성을 남긴다.
- MVO cap 40%는 방법론의 부가 장치가 아니라 small-sample MVO를 안정화하는 핵심 제약이다.

## III. 실험설계

이 장의 역할은 "결과를 어떻게 믿을 것인가"를 설명하는 것이다. 데이터, baseline, ablation, 평가 지표를 명확히 정리해 결과 분석의 기준을 먼저 세운다.

장 전체 흐름:

```text
데이터와 기간 정의
-> 분류 모델 비교 설계
-> 포트폴리오 전략 비교 설계
-> 평가 지표 정의
```

### Slide 11. 데이터와 Train/Test 구성

한 장의 메시지:

> 동일한 데이터와 기간에서 classification 성능과 portfolio downstream 성과를 함께 평가한다.

세부 목차:

- 자산 universe
  - SPY: 미국 대형주
  - QQQ: 성장주/기술주
  - GLD: 금
  - TLT: 장기채
- 입력 데이터
  - OHLCV 기반 feature
  - cross-asset supervised dataset
  - 30거래일 sequence
- 예측 목표
  - 5거래일 후 Bear vs Non-Bear
  - binary soft label 사용
- test 구간
  - 2024-04-15 ~ 2026-05-15
- 사용 dataset
  - `cross_asset_supervised_30d_5d_binary_soft_labels`

발표 포인트:

- 주식, 채권, 금을 함께 넣어 regime별 자산 역할 변화를 반영한다.
- test 구간 하나의 결과이므로 이후 한계에서 일반화 문제를 다시 언급한다.

### Slide 12. 분류 실험 설계: Baseline과 Ablation

한 장의 메시지:

> 딥러닝 모델과 soft label의 효과를 보이기 위해 classical baseline과 label ablation을 함께 비교한다.

세부 목차:

- baseline 비교 목적
  - 작은 샘플에서 딥러닝이 정말 필요한지 확인
  - 단순 모델보다 나은지 검증
- 비교 모델
  - Logistic Regression
  - Random Forest
  - Binary Conv1D+LSTM hard label
  - Binary Conv1D+LSTM soft label
- ablation 질문
  - 3-class에서 binary로 바꾸면 개선되는가?
  - hard label에서 soft label로 바꾸면 Bear detection이 개선되는가?
  - LR/RF보다 Conv1D+LSTM이 나은가?

비교 표:

| 모델 | 목적 |
|---|---|
| Logistic Regression | 선형 baseline |
| Random Forest | 비선형 classical baseline |
| Conv1D+LSTM Binary Hard | hard label 딥러닝 기준 |
| Conv1D+LSTM Binary Soft | 최종 국면 예측 모델 |

발표 포인트:

- "딥러닝을 썼다"가 아니라 "baseline보다 나은지 확인했다"는 점이 중요하다.
- 결과 파트에서 Bear Recall과 Balanced Accuracy를 중심으로 비교한다.

### Slide 13. 포트폴리오 실험 설계: 전략과 평가 지표

한 장의 메시지:

> 최종 전략은 강한 단순 benchmark 및 기존 3-class 전략과 비교해야 한다.

세부 목차:

- 정적 benchmark
  - 60/40
  - Buy & Hold
  - EW 1/N
- 동적 전략
  - 3-class Regime-MVO cap 40%
  - Binary Soft 2-Regime MVO cap 40%
- MVO cap 실험
  - cap 없음: 극단적 비중 확인
  - cap 50%: 부분 완화
  - cap 40%: 최종 선택
- 평가 지표
  - 누적수익률: 수익성
  - Sharpe: 위험 대비 수익
  - MDD: 최악 손실 구간
  - Calmar: drawdown 대비 성과

전략 비교 표:

| 전략 | 역할 |
|---|---|
| 60/40 | 전통적 정적 benchmark |
| Buy & Hold | 위험자산 보유 benchmark |
| EW 1/N | 강한 단순 분산 benchmark |
| 3-class Regime-MVO cap 40% | 초기 regime-aware 전략 |
| Binary Soft 2-Regime MVO cap 40% | 최종 전략 |

발표 포인트:

- EW 1/N은 단순하지만 강한 benchmark이므로 최종 전략과 반드시 비교한다.
- 최종 전략은 수익률만이 아니라 drawdown과 Calmar까지 같이 봐야 한다.

## IV. 결과 분석, 한계와 결론

이 장의 역할은 "그래서 실제로 무엇이 좋아졌는가"와 "그 결과를 어디까지 믿어야 하는가"를 균형 있게 말하는 것이다. 과장하지 않고, 개선된 부분과 한계를 동시에 제시한다.

장 전체 흐름:

```text
분류 결과
-> Soft label의 Bear Recall 개선
-> MVO cap의 몰빵 완화
-> 최종 backtest 성과
-> 한계와 향후 개선
-> 최종 결론
```

### Slide 14. 분류 결과: Binary Soft Label의 개선

Figure:

- [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png)
- [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png)

한 장의 메시지:

> Binary soft label은 hard label보다 Bear Recall을 개선했고, LR/RF baseline보다 좋은 분류 성능을 보였다.

세부 목차:

- 3-class 문제
  - Balanced Accuracy 51.9%
  - Neutral Recall 0.0%
- binary hard label 개선
  - Balanced Accuracy 70.2%
  - Bear Recall 58.1%
- binary soft label 개선
  - Balanced Accuracy 72.4%
  - Bear Recall 67.4%
- baseline 대비
  - LR Bear Recall 32.6%
  - RF Bear Recall 53.5%
  - Binary soft label이 Bear detection에서 가장 우수

결과 표:

| 모델 | Balanced Accuracy | Bear Recall |
|---|---:|---:|
| 3-class hard label | 51.9% | 60.5% |
| LR baseline | 61.4% | 32.6% |
| RF baseline | 66.3% | 53.5% |
| Binary hard label | 70.2% | 58.1% |
| Binary soft label | 72.4% | 67.4% |

발표 포인트:

- binary soft label은 단순히 전체 accuracy만 개선한 것이 아니라 Bear Recall을 개선했다.
- 하방 위험 관리 목적에서는 Bear Recall 개선이 특히 중요하다.
- 분류 성능이 곧바로 투자 성과를 보장하지 않으므로 다음 슬라이드에서 portfolio layer를 확인한다.

### Slide 15. MVO Cap 효과: 극단적 비중 완화

Figure:

- [fig09_binary_mvo_weights.png](../outputs/figures/final/fig09_binary_mvo_weights.png)

한 장의 메시지:

> 제약 없는 MVO는 특정 자산에 몰빵하지만, cap 40%는 추정 오차에 대한 민감도를 줄인다.

세부 목차:

- 문제
  - MVO는 기대수익률과 공분산 추정에 민감
  - regime별로 데이터를 나누면 sample 수가 더 작아짐
  - 작은 추정 오차가 극단적 weight로 이어질 수 있음
- cap 없음
  - Non-Bear: SPY 100%
  - Bear: TLT 100%
- cap 적용
  - 50% cap: 몰빵 완화
  - 40% cap: 더 분산된 비중
- 최종 선택
  - cap 40%를 최종 전략에 적용

결과 표:

| Cap | Non-Bear MVO | Bear MVO |
|---:|---|---|
| 100% | SPY 100.0% | TLT 100.0% |
| 50% | SPY 50.0%, QQQ 49.8%, GLD 0.2% | QQQ 7.3%, GLD 42.7%, TLT 50.0% |
| 40% | SPY 40.0%, QQQ 40.0%, GLD 17.3%, TLT 2.7% | QQQ 20.0%, GLD 40.0%, TLT 40.0% |

발표 포인트:

- cap은 성과를 억지로 맞추는 장치가 아니라 추정 오차를 줄이기 위한 안정화 제약이다.
- MVO의 이론적 최적해가 현실에서는 너무 공격적인 비중을 만들 수 있음을 보여준다.

### Slide 16. 최종 Backtest 결과

Figure:

- [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png)
- [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png)
- [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png)

한 장의 메시지:

> Binary Soft 2-Regime MVO cap 40%는 현재 비교군 중 누적수익률, Sharpe, Calmar 기준으로 가장 균형 잡힌 결과를 냈다.

세부 목차:

- 최종 전략
  - Binary soft label model
  - 2-Regime MVO
  - cap 40%
- 주요 benchmark와 비교
  - 60/40보다 수익성과 위험 대비 성과 우수
  - Buy & Hold보다 MDD가 낮음
  - EW 1/N보다 누적수익률과 Calmar가 높고 MDD는 유사
  - 3-class capped MVO보다 소폭 개선
- 해석
  - binary 재정의와 soft label이 downstream portfolio 성과에도 연결됨
  - 다만 차이가 압도적으로 크다고 과장하지 않고 "균형 잡힌 개선"으로 표현

결과 표:

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

발표 포인트:

- EW 1/N 대비 누적수익률은 +2.8%p 높고 MDD는 거의 비슷하다.
- Buy & Hold 대비 누적수익률은 조금 높지만 MDD는 크게 낮다.
- 최종 결론은 "무조건 수익률 극대화"가 아니라 "risk-return 균형 개선"이다.

### Slide 17. 한계와 결론

한 장의 메시지:

> 본 연구는 국면 확률을 포트폴리오 의사결정에 연결할 수 있음을 보였지만, pseudo-label과 단일 test 구간이라는 한계가 있다.

세부 목차:

- 결론 1: 문제 정의
  - Neutral은 구조적으로 애매했고, Bear vs Non-Bear binary task가 더 안정적이었다.
- 결론 2: 예측 모델
  - LR/RF baseline보다 Conv1D+LSTM이 Bear detection에서 더 나았다.
- 결론 3: soft label
  - HMM posterior의 불확실성을 반영해 Bear Recall을 개선했다.
- 결론 4: portfolio layer
  - 2-Regime MVO와 weight cap 40%를 결합해 극단적 비중을 완화했다.
- 결론 5: 최종 성과
  - Binary Soft Label + 2-Regime MVO + cap 40%가 현재 비교군 중 가장 균형 잡힌 결과를 냈다.

한계:

- HMM label은 실제 정답이 아니라 pseudo-label이다.
- sample 수가 작고 test 구간이 하나다.
- cap 40%는 현재 test 구간에서 좋은 값이며, 엄밀하게는 validation/walk-forward 방식으로 선택해야 한다.
- 거래비용, 세금, 시장충격은 충분히 반영되지 않았다.
- cash 또는 short-term bond를 추가하면 방어 성과가 달라질 수 있다.

향후 개선:

- walk-forward validation으로 cap과 hyperparameter 선택
- covariance shrinkage 또는 robust MVO 적용
- transaction cost를 포함한 현실적 backtest
- 현금/단기채 등 방어 자산 universe 확장
- regime label 자체를 다른 방법과 비교

마지막 문장:

> 따라서 본 프로젝트는 "시장 국면을 정확히 맞히는 모델"보다 "위험 국면 확률을 포트폴리오 의사결정에 연결하는 구조"에 초점을 둔 실험이다.

## 3. PPT 제작용 압축 목차

실제 PPT에는 아래처럼 장별 첫 슬라이드에 세부 목차를 넣으면 발표 흐름이 선명해진다.

### I. 연구 배경, 선행연구

1. 제목 및 연구 질문
2. 정적 포트폴리오의 한계
3. 예측을 투자 의사결정으로 연결
4. 선행연구와 본 연구의 차별점

### II. 방법론

1. 전체 방법론 파이프라인
2. HMM pseudo-label 생성
3. 3-class 문제 진단: Neutral label failure
4. Binary Bear Detection으로 재정의
5. Conv1D+LSTM 국면 예측 모델
6. Binary Soft Label과 2-Regime MVO

### III. 실험설계

1. 데이터와 Train/Test 구성
2. 분류 실험 설계: Baseline과 Ablation
3. 포트폴리오 실험 설계: 전략과 평가 지표

### IV. 결과 분석, 한계와 결론

1. 분류 결과: Binary Soft Label의 개선
2. MVO Cap 효과: 극단적 비중 완화
3. 최종 Backtest 결과
4. 한계와 결론

## 4. 12장 압축 발표 버전

시간이 10-12분으로 줄어들면 아래 12장 구성으로 압축한다.

| 압축 슬라이드 | 원래 슬라이드 | 제목 |
|---:|---|---|
| 1 | 1 | 제목 및 연구 질문 |
| 2 | 2-3 | 정적 포트폴리오의 한계와 연구 목적 |
| 3 | 4 | 선행연구와 차별점 |
| 4 | 5 | 전체 파이프라인 |
| 5 | 6-8 | HMM pseudo-label, Neutral 실패, Binary 재정의 |
| 6 | 9-10 | Conv1D+LSTM, Soft Label, 2-Regime MVO |
| 7 | 11 | 데이터와 Train/Test 구성 |
| 8 | 12-13 | Baseline, 전략, 평가 지표 |
| 9 | 14 | 분류 결과 |
| 10 | 15 | MVO Cap 효과 |
| 11 | 16 | 최종 Backtest 결과 |
| 12 | 17 | 한계와 결론 |

## 5. 사용할 최종 Figure

발표/보고서에는 `outputs/figures/final/` 아래 그림만 사용한다.

| 그림 | 파일 | 역할 | 추천 슬라이드 |
|---|---|---|---:|
| Fig 01 | [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png) | 전체 파이프라인 | 5 |
| Fig 02 | [fig02_related_work.png](../outputs/figures/final/fig02_related_work.png) | 선행연구 비교 | 4 |
| Fig 03-A | [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png) | 최종 전략 누적수익률 / drawdown 경로 | 16 |
| Fig 03-B | [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png) | 최종 성과 요약 | 16 |
| Fig 04 | [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png) | 분류 성능 개선 | 14 |
| Fig 05 | [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png) | Binary soft label confusion matrix | 14 |
| Fig 07 | [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png) | Ablation / benchmark | 16 |
| Fig 09 | [fig09_binary_mvo_weights.png](../outputs/figures/final/fig09_binary_mvo_weights.png) | MVO cap 효과 | 15 |

## 6. PPT 제작 순서 추천

1. Slide 5의 전체 파이프라인을 먼저 만든다.
2. Slide 7-8에서 "왜 3-class가 아니라 binary인가"를 설득한다.
3. Slide 10에서 soft label과 MVO 결합 수식을 깔끔하게 정리한다.
4. Slide 14-16의 결과 슬라이드를 먼저 완성한다.
5. 마지막으로 Slide 2-4의 배경과 선행연구를 앞에 붙인다.

이 순서로 만들면 발표의 핵심 스토리인 "문제 정의 변경 -> soft label -> capped MVO -> 최종 성과"가 흐트러지지 않는다.

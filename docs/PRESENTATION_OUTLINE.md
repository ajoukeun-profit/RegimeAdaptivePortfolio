# 발표 목차: 시장 국면 인식 기반 동적 포트폴리오 전략

> 영문 부제: Market Regime-Aware Dynamic Portfolio Strategy

## 0. 발표 전체 메시지

본 프로젝트는 가격 자체를 직접 예측하는 대신, 시장이 위험 국면인지 아닌지를 확률적으로 예측하고, 그 예측 확률을 포트폴리오 비중 조절에 연결하는 동적 자산배분 전략을 검증한다.

최종 전략:

```text
Binary Soft Label + 2-Regime MVO + Weight Cap 40%
```

전체 스토리:

```text
정적 포트폴리오의 한계
-> 시장 국면 인식 필요
-> SPY/QQQ/GLD/TLT cross-asset 데이터 구성
-> HMM으로 Bear / Neutral / Bull pseudo-label 생성
-> Conv1D+LSTM으로 미래 Bear 확률 예측
-> 실험 중 Neutral label failure 확인
-> Bear vs Non-Bear binary task로 재정의
-> HMM posterior를 binary soft label로 활용
-> 2-Regime MVO와 weight cap 40%로 포트폴리오 비중 산출
-> 최종 backtest 성과 평가
```

핵심 주장:

1. 금융시장은 국면별로 자산의 위험, 상관관계, 방어 역할이 달라지므로 고정 비중에는 한계가 있다.
2. HMM label은 실제 정답이 아니라 pseudo-label이므로, 분류 성능은 포트폴리오 성과와 함께 해석해야 한다.
3. 3-class의 Neutral은 경제적 경계가 애매했고, test에서 한 번도 예측되지 않았다.
4. 투자 목적상 모든 국면을 세밀하게 맞히는 것보다 Bear를 탐지하는 것이 더 중요하다.
5. Binary soft label은 HMM posterior의 불확실성을 보존하면서 Bear Recall을 개선했다.
6. 예측 확률을 2-Regime MVO에 연결하고 cap 40%를 적용했을 때 현재 비교군 중 가장 균형 잡힌 risk-return 결과를 얻었다.

최종 결과:

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

마지막 결론 문장:

> 이 프로젝트의 핵심은 시장을 완벽히 예측하는 것이 아니라, 위험 국면의 확률을 인식하고 그 확률을 포트폴리오 비중 조절에 연결하는 것이다.

## 1. PPT 목차와 담당자

이번 발표는 담당자는 4명으로 나누되, PPT 목차에서는 `연구 배경`과 `선행 연구`를 분리해 5개 큰 목차로 보여주는 편이 가장 깔끔하다. 준현 파트 안에서 배경과 선행연구를 이어서 발표하더라도, 목차상으로는 두 항목을 분리하면 청중이 "문제 제기 -> 기존 연구 -> 우리 방법" 흐름을 더 쉽게 따라갈 수 있다.

PPT 목차 추천:

| 순서 | 한글 목차 | English Title | 담당 | 핵심 질문 | 슬라이드 |
|---:|---|---|---|---|---:|
| 1 | 연구 배경 | Research Background | 준현 | 왜 국면 인식 포트폴리오가 필요한가? | 1-3 |
| 2 | 선행 연구 | Related Work | 준현 | 기존 연구와 본 프로젝트의 차이는 무엇인가? | 4 |
| 3 | 데이터 및 방법론 | Data and Methodology | 재현 | 어떤 데이터로 어떤 전략을 설계했는가? | 5-10 |
| 4 | 실험 설계 및 트러블슈팅 | Experimental Design and Troubleshooting | 예림 | 어떤 비교 실험을 했고 어떤 문제를 해결했는가? | 11-15 |
| 5 | 결과 분석, 한계와 결론 | Results, Limitations, and Conclusion | 준한 | 실제로 좋아졌는가, 어디까지 믿을 수 있는가? | 16-18 |

총 18장 기준이다. 15분 발표에서는 Slide 12-15의 트러블슈팅 파트를 빠르게 지나가고, Slide 16-17 결과 파트에 시간을 조금 더 준다.

## 2. 장별 세부 목차

## I. 연구 배경 / Research Background

이 장의 역할은 "왜 이 프로젝트를 해야 하는가"를 설득하는 것이다. 기술 소개보다 먼저 정적 포트폴리오의 한계와 시장 국면 인식의 필요성을 제시한다.

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
- 핵심 질문: 시장 상태가 바뀔 때 포트폴리오도 바뀌어야 하는가?
- 접근 방식: 국면 예측 확률을 자산배분 비중에 연결
- 사용 자산: SPY, QQQ, GLD, TLT
- 최종 키워드: HMM, Conv1D+LSTM, Binary Soft Label, MVO

왜 이 주제를 선택했나:

- 금융 딥러닝 프로젝트에서 예측 모델만 만들면 투자 의사결정과의 연결이 약하다.
- 국면 확률을 실제 포트폴리오 비중으로 연결하면 모델의 경제적 의미를 평가할 수 있다.

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

왜 국면 인식이 필요한가:

- 같은 자산도 상승장과 하락장에서 위험 기여도가 달라질 수 있다.
- 따라서 포트폴리오가 시장 환경 변화에 반응할 수 있어야 한다.

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

왜 Bear Recall을 보는가:

- 투자 관점에서 가장 치명적인 문제는 상승장을 놓치는 것보다 하락 위험을 놓치는 것이다.
- 따라서 단순 accuracy보다 Bear Recall이 더 중요한 보조 지표가 된다.

발표 포인트:

- 분류 정확도는 중간 지표이고 최종 목적은 포트폴리오 성과다.
- 이후 결과 분석에서도 classification과 portfolio 성과를 함께 본다.

## II. 선행 연구 / Related Work

이 장의 역할은 기존 연구 흐름을 짚고, 본 프로젝트가 어디에 위치하는지 설명하는 것이다. 준현이 연구 배경에 이어 발표하면 자연스럽다.

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
- 이 연결 구조가 이후 데이터 및 방법론 파트의 중심이다.

## III. 데이터 및 방법론 / Data and Methodology

이 장의 역할은 "어떤 데이터로 어떤 전략을 설계했는가"를 설명하는 것이다. 여기서는 트러블슈팅 결과를 길게 말하지 않고, 최종 전략을 구성하는 데이터, label, 모델, 포트폴리오 비중 산출 방식을 차분히 설명한다.

장 전체 흐름:

```text
자산 universe 선정
-> OHLCV 및 cross-asset feature 구성
-> HMM pseudo-label 생성
-> Binary soft label 정의
-> Conv1D+LSTM 확률 예측
-> 2-Regime MVO와 weight cap
```

### Slide 5. 데이터 Universe와 선택 이유

한 장의 메시지:

> 주식, 성장주, 금, 장기채를 함께 사용해 국면별 자산 역할 변화를 반영한다.

세부 목차:

- 사용 자산
  - SPY: 미국 대형주/시장 대표 ETF
  - QQQ: 성장주 및 기술주 성격이 강한 ETF
  - GLD: 금, 대체자산 및 위험회피 자산 후보
  - TLT: 미국 장기채, 금리 및 위험회피 국면 반응 자산
- 사용 데이터
  - daily OHLCV
  - 4개 자산을 같은 날짜 기준으로 정렬
  - 결측 및 거래일 정합성 처리
- 분석 의도
  - 주식형 위험자산과 방어/대체 자산을 함께 관찰
  - 단일 자산 예측이 아니라 cross-asset regime signal을 학습

왜 이 자산들을 선택했나:

- SPY와 QQQ는 위험자산 내부의 시장/성장주 성격 차이를 반영한다.
- GLD와 TLT는 하락장 또는 위험회피 국면에서 포트폴리오 방어 역할을 할 수 있다.
- 네 자산 모두 ETF라 가격 자료 접근성이 좋고 backtest 해석이 직관적이다.

발표 포인트:

- 이 프로젝트의 입력은 SPY 하나가 아니라 여러 자산의 상대 움직임이다.
- 국면 인식은 자산 간 관계 변화를 보는 문제이므로 cross-asset 구성이 중요하다.

### Slide 6. Feature, Window, Label 구성

한 장의 메시지:

> 최근 30거래일의 cross-asset feature로 5거래일 후 시장 국면을 예측하는 supervised dataset을 만든다.

세부 목차:

- 입력 window
  - 최근 30거래일
  - 시계열 순서를 유지한 sequence 형태
- feature 구성
  - OHLCV 기반 feature
  - 수익률, 변동성, 이동평균/상대 변화 등 cross-asset 정보
  - 최종 입력 형태: 30 days x 40 features
- 예측 horizon
  - 5거래일 후 regime
- 최종 dataset
  - `cross_asset_supervised_30d_5d_binary_soft_labels`
- 평가 구간
  - test 기간: 2024-04-15 ~ 2026-05-15

왜 30일 window와 5일 horizon인가:

- 30거래일은 약 1.5개월 정도의 단기-중기 시장 흐름을 담는다.
- 5거래일 horizon은 너무 짧은 일간 noise보다 한 주 단위의 국면 변화를 보려는 설정이다.

발표 포인트:

- feature는 단일 시점 값이 아니라 30일 sequence다.
- 따라서 뒤에서 Conv1D+LSTM을 쓰는 이유가 자연스럽게 연결된다.

### Slide 7. HMM Pseudo-label 생성

한 장의 메시지:

> HMM은 관측되지 않는 시장 상태를 추정하지만, 그 결과는 실제 정답이 아니라 모델이 만든 pseudo-label이다.

세부 목차:

- HMM의 역할
  - 관측 데이터에서 잠재 시장 상태 추정
  - 상태를 Bear / Neutral / Bull로 사후 해석
- pseudo-label의 의미
  - 사람이 직접 labeling한 정답이 아님
  - HMM의 통계적 추정 결과
  - classification 성능을 실제 시장 예측 정확도로 과대해석하면 안 됨
- posterior probability의 활용
  - hard label: 가장 확률이 높은 state만 사용
  - soft label: 각 state 확률을 학습 target으로 활용

수식:

```text
s_t in {Bear, Neutral, Bull}
f_theta(X_t) ~= y_hat_{t+h}^{HMM}
```

왜 HMM을 선택했나:

- 시장 국면은 사람이 직접 관측할 수 없는 잠재 상태다.
- HMM은 이런 latent state를 시계열 자료에서 추정하는 데 적합하다.
- 직접 label이 없는 상황에서 supervised learning을 위한 pseudo-label을 만들 수 있다.

발표 포인트:

- HMM label을 "정답"이라고 부르지 않고 "pseudo-label"이라고 계속 표현한다.
- 이 점이 soft label로 넘어가는 논리적 근거다.

### Slide 8. Binary Soft Label 정의

한 장의 메시지:

> 최종 학습 target은 Bear vs Non-Bear이며, HMM posterior를 합쳐 binary soft label로 사용한다.

세부 목차:

- 초기 HMM state

```text
Bear / Neutral / Bull
```

- 최종 binary state

```text
Bear / Non-Bear
Non-Bear = Neutral + Bull
```

- soft target

```text
P(Bear) = P(Bear)
P(Non-Bear) = P(Neutral) + P(Bull)
```

- loss 관점
  - hard label: 한 class만 정답으로 취급
  - soft label: target probability distribution을 학습

왜 binary soft label을 선택했나:

- 투자 목적은 Neutral/Bull을 세밀하게 구분하는 것보다 Bear를 탐지하는 데 가깝다.
- HMM posterior의 불확실성을 hard label로 버리지 않고 학습에 반영할 수 있다.
- binary 확률은 이후 2-Regime MVO와 직접 연결된다.

발표 포인트:

- 이 슬라이드에서는 "실패해서 바꿨다"보다 "최종 target을 이렇게 정의한다"를 중심으로 말한다.
- 실제로 왜 3-class가 문제가 됐는지는 실험설계 및 트러블슈팅 장에서 자세히 보여준다.

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
- 학습 방식
  - soft cross entropy
  - AdamW optimizer
  - dropout, early stopping, gradient clipping
- 출력의 의미
  - class label만이 아니라 portfolio weight에 들어가는 probability

왜 Conv1D+LSTM을 선택했나:

- Conv1D는 최근 며칠의 국소 패턴을 잡는 데 유리하다.
- LSTM은 30일 sequence의 시간적 누적 흐름을 반영할 수 있다.
- 단순 tabular 모델보다 시계열 순서를 활용할 수 있다.

발표 포인트:

- 모델의 출력값은 "맞다/틀리다" 판단으로 끝나지 않고 MVO 비중 혼합에 사용된다.
- 그래서 확률의 품질이 downstream 성과와 연결된다.

### Slide 10. 2-Regime MVO와 Weight Cap

한 장의 메시지:

> 예측된 Bear 확률을 이용해 Non-Bear MVO와 Bear MVO 비중을 가중 평균한다.

세부 목차:

- 2-Regime MVO
  - Non-Bear 구간의 MVO 비중 계산
  - Bear 구간의 MVO 비중 계산
  - 테스트 시점마다 예측 확률로 두 비중을 혼합
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

왜 2-Regime MVO를 선택했나:

- 모델이 직접 자산 비중을 예측하게 하면 해석이 어렵고 과적합 위험이 크다.
- MVO는 각 regime에서의 평균-분산 구조를 반영해 비중을 계산한다.
- 예측 확률로 두 regime 비중을 섞으면 hard switching보다 부드러운 동적 배분이 가능하다.

왜 weight cap을 넣었나:

- regime별 sample이 작아지면 MVO가 특정 자산에 과도하게 몰빵하기 쉽다.
- cap 40%는 이 추정 오차 민감도를 줄이기 위한 안정화 제약이다.

발표 포인트:

- MVO cap 40%는 방법론의 부가 장치가 아니라 small-sample MVO를 안정화하는 핵심 제약이다.
- cap 값 선택의 엄밀성은 한계에서 다시 언급한다.

## IV. 실험 설계 및 트러블슈팅 / Experimental Design and Troubleshooting

이 장의 역할은 "어떤 비교 실험을 했고, 어떤 문제를 확인해서 최종 구조로 갔는가"를 설명하는 것이다. 여기서 3-class 실패, 모델 학습 안정화, baseline 비교, MVO cap 문제를 한꺼번에 정리한다.

장 전체 흐름:

```text
Train/Test 및 평가 지표 정의
-> 3-class Neutral failure 확인
-> 학습 안정화 점검
-> LR/RF baseline 비교
-> binary hard / soft label 비교
-> MVO cap 실험
```

### Slide 11. Train/Test와 평가 지표

한 장의 메시지:

> 동일한 데이터와 기간에서 classification 성능과 portfolio downstream 성과를 함께 평가한다.

세부 목차:

- test 구간
  - 2024-04-15 ~ 2026-05-15
- classification 평가
  - Accuracy
  - Balanced Accuracy
  - Macro F1
  - Bear Recall
- portfolio 평가
  - Cumulative Return
  - Sharpe
  - MDD
  - Calmar
- 비교 단위
  - label 정의별 성능 비교
  - model baseline 비교
  - portfolio strategy 비교

왜 두 종류의 지표를 같이 보는가:

- 분류 성능이 좋아도 포트폴리오 성과로 이어지지 않을 수 있다.
- 이 프로젝트의 최종 목표는 모델 성능표가 아니라 투자 전략의 risk-return 개선이다.

발표 포인트:

- Balanced Accuracy는 class imbalance를 고려하기 위해 사용한다.
- Bear Recall은 하방 위험 탐지 능력을 보기 위한 핵심 보조 지표다.

### Slide 12. 트러블슈팅 1: 3-class Neutral Label Failure

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
- 시도한 대응
  - class weight 조정
  - `neutral_boost` 적용
- 최종 판단
  - class weight 조정보다 Bear vs Non-Bear 문제 재정의가 더 적절

결과 표:

| 지표 | 값 |
|---|---:|
| Accuracy | 61.9% |
| Balanced Accuracy | 51.9% |
| Bear Recall | 60.5% |
| Neutral Recall | 0.0% |
| Bull Recall | 95.1% |

발표 포인트:

- 이 슬라이드는 최종 binary task로 넘어가는 가장 중요한 근거다.
- "모델이 나쁘다"가 아니라 "Neutral이라는 target 자체가 투자 목적에 덜 맞다"로 설명한다.

### Slide 13. 트러블슈팅 2: 모델 학습 안정화 점검

한 장의 메시지:

> 학습 불안정성도 점검했지만, 최종 실패 원인은 gradient 문제보다 label 정의와 MVO 추정 안정성에 가까웠다.

세부 목차:

- 점검한 문제
  - gradient explosion 가능성
  - gradient vanishing 가능성
  - loss divergence 또는 NaN 발생 여부
  - class imbalance로 인한 특정 class 무시
- 적용한 학습 안정화
  - gradient clipping
  - dropout
  - early stopping
  - AdamW
  - class weight 및 `neutral_boost`
- 관찰 결과
  - loss divergence나 NaN 증거는 없음
  - 30일 sequence라 심각한 vanishing gradient 가능성은 상대적으로 낮음
  - Neutral failure는 학습 안정성보다 label ambiguity 문제로 해석

관련 코드 포인트:

```python
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

왜 이 내용을 넣는가:

- 발표에서 "모델이 못 배운 것 아닌가?"라는 질문이 나올 수 있다.
- gradient 문제도 점검했지만 핵심 병목은 label definition과 portfolio optimization이었다는 점을 보여준다.

발표 포인트:

- gradient 문제를 완전히 배제하는 것은 아니지만, 관찰된 실패 양상은 Neutral label ambiguity가 훨씬 명확했다.
- 이 점이 binary 재정의의 설득력을 높인다.

### Slide 14. 실험설계: Baseline과 Label Ablation

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

왜 baseline을 넣었나:

- 데이터 수가 크지 않으므로 딥러닝이 항상 유리하다고 가정하면 안 된다.
- LR/RF보다 좋아야 Conv1D+LSTM 사용의 설득력이 생긴다.

발표 포인트:

- 결과 파트에서 Bear Recall과 Balanced Accuracy를 중심으로 비교한다.
- "딥러닝을 썼다"가 아니라 "baseline보다 나은지 확인했다"는 점이 중요하다.

### Slide 15. 트러블슈팅 3: MVO 추정 오차와 Weight Cap

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

왜 cap을 선택했나:

- cap은 성과를 억지로 맞추는 장치가 아니라 추정 오차를 줄이기 위한 안정화 제약이다.
- MVO의 이론적 최적해가 현실에서는 너무 공격적인 비중을 만들 수 있다.

발표 포인트:

- 이 슬라이드는 포트폴리오 layer의 트러블슈팅이다.
- cap 값 선택은 현재 test 구간 기준이므로 한계에서 walk-forward validation 필요성을 언급한다.

## V. 결과 분석, 한계와 결론 / Results, Limitations, and Conclusion

이 장의 역할은 "그래서 실제로 무엇이 좋아졌는가"와 "그 결과를 어디까지 믿어야 하는가"를 균형 있게 말하는 것이다. 과장하지 않고, 개선된 부분과 한계를 동시에 제시한다.

장 전체 흐름:

```text
분류 결과
-> Binary soft label의 Bear Recall 개선
-> 최종 backtest 성과
-> 한계와 향후 개선
-> 최종 결론
```

### Slide 16. 분류 결과: Binary Soft Label의 개선

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

해석:

- binary soft label은 단순히 전체 accuracy만 개선한 것이 아니라 Bear Recall을 개선했다.
- 하방 위험 관리 목적에서는 Bear Recall 개선이 특히 중요하다.
- 분류 성능이 곧바로 투자 성과를 보장하지 않으므로 다음 슬라이드에서 portfolio 결과를 확인한다.

발표 포인트:

- 여기서는 Slide 12-14의 트러블슈팅이 실제 성능 개선으로 이어졌는지 보여준다.
- LR < RF < LSTM 흐름과 hard < soft 흐름을 함께 강조한다.

### Slide 17. 최종 Backtest 결과

Figure:

- [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png)
- [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png)
- [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png)

한 장의 메시지:

> Binary Soft 2-Regime MVO cap 40%는 현재 비교군 중 누적수익률, Sharpe, Calmar 기준으로 가장 균형 잡힌 결과를 냈다.

세부 목차:

- 비교 전략
  - 60/40
  - Buy & Hold
  - EW 1/N
  - 3-class Regime-MVO cap 40%
  - Binary Regime-MVO Soft cap 40%
- 주요 결과
  - 60/40보다 수익성과 위험 대비 성과 우수
  - Buy & Hold보다 MDD가 낮음
  - EW 1/N보다 누적수익률과 Calmar가 높고 MDD는 유사
  - 3-class capped MVO보다 소폭 개선
- 해석
  - binary 재정의와 soft label이 downstream portfolio 성과에도 연결됨
  - cap 40%가 MVO extreme weight 문제를 완화
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

### Slide 18. 한계와 결론

한 장의 메시지:

> 본 연구는 국면 확률을 포트폴리오 의사결정에 연결할 수 있음을 보였지만, pseudo-label과 단일 test 구간이라는 한계가 있다.

결론:

1. 정적 포트폴리오의 한계를 보완하기 위해 시장 국면 확률을 활용했다.
2. HMM pseudo-label을 만들고, Conv1D+LSTM으로 Bear 확률을 예측했다.
3. 3-class Neutral failure를 확인한 뒤, 투자 목적에 맞게 Bear vs Non-Bear로 재정의했다.
4. 모델 학습 안정화와 baseline 비교를 통해 gradient 문제보다 label definition 문제가 핵심임을 확인했다.
5. Binary soft label은 HMM posterior의 불확실성을 반영해 Bear Recall을 개선했다.
6. MVO weight cap 40%는 작은 샘플에서 발생하는 극단적 비중을 완화했다.
7. 최종적으로 Binary Soft Label + 2-Regime MVO + cap 40%가 현재 비교군 중 가장 균형 잡힌 결과를 냈다.

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
- HMM 외 다른 regime labeling 방법과 비교

마지막 문장:

> 따라서 본 프로젝트는 "시장 국면을 정확히 맞히는 모델"보다 "위험 국면 확률을 포트폴리오 의사결정에 연결하는 구조"에 초점을 둔 실험이다.

## 3. PPT 제작용 압축 목차

실제 PPT에는 아래처럼 장별 첫 슬라이드에 세부 목차를 넣으면 발표 흐름이 선명해진다.

### I. 연구 배경 / Research Background

1. 제목 및 연구 질문
2. 정적 포트폴리오의 한계
3. 예측을 투자 의사결정으로 연결

### II. 선행 연구 / Related Work

1. 관련 연구 흐름
2. 본 프로젝트의 차별점

### III. 데이터 및 방법론 / Data and Methodology

1. 데이터 universe와 선택 이유
2. Feature, window, label 구성
3. HMM pseudo-label 생성
4. Binary soft label 정의
5. Conv1D+LSTM 국면 예측 모델
6. 2-Regime MVO와 weight cap

### IV. 실험 설계 및 트러블슈팅 / Experimental Design and Troubleshooting

1. Train/Test와 평가 지표
2. 트러블슈팅 1: 3-class Neutral label failure
3. 트러블슈팅 2: 모델 학습 안정화 점검
4. 실험설계: baseline과 label ablation
5. 트러블슈팅 3: MVO 추정 오차와 weight cap

### V. 결과 분석, 한계와 결론 / Results, Limitations, and Conclusion

1. 분류 결과: Binary soft label의 개선
2. 최종 backtest 결과
3. 한계와 결론

## 4. 12장 압축 발표 버전

시간이 10-12분으로 줄어들면 아래 12장 구성으로 압축한다.

| 압축 슬라이드 | 원래 슬라이드 | 제목 |
|---:|---|---|
| 1 | 1 | 제목 및 연구 질문 |
| 2 | 2-3 | 정적 포트폴리오의 한계와 연구 목적 |
| 3 | 4 | 선행연구와 차별점 |
| 4 | 5-6 | 데이터 universe와 feature/label 구성 |
| 5 | 7-8 | HMM pseudo-label과 binary soft label |
| 6 | 9-10 | Conv1D+LSTM, 2-Regime MVO, cap 40% |
| 7 | 11 | Train/Test와 평가 지표 |
| 8 | 12-13 | Neutral failure와 모델 학습 안정화 |
| 9 | 14-15 | Baseline, label ablation, MVO cap |
| 10 | 16 | 분류 결과 |
| 11 | 17 | 최종 backtest 결과 |
| 12 | 18 | 한계와 결론 |

## 5. 사용할 최종 Figure

발표/보고서에는 `outputs/figures/final/` 아래 그림만 사용한다.

| 그림 | 파일 | 역할 | 추천 슬라이드 |
|---|---|---|---:|
| Fig 01 | [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png) | 전체 파이프라인 | 5 |
| Fig 02 | [fig02_related_work.png](../outputs/figures/final/fig02_related_work.png) | 선행연구 비교 | 4 |
| Fig 03-A | [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png) | 최종 전략 누적수익률 / drawdown 경로 | 17 |
| Fig 03-B | [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png) | 최종 성과 요약 | 17 |
| Fig 04 | [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png) | 분류 성능 개선 | 16 |
| Fig 05 | [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png) | Binary soft label confusion matrix | 16 |
| Fig 07 | [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png) | Ablation / benchmark | 17 |
| Fig 09 | [fig09_binary_mvo_weights.png](../outputs/figures/final/fig09_binary_mvo_weights.png) | MVO cap 효과 | 15 |

## 6. PPT 제작 순서 추천

1. Slide 5-6의 데이터 및 feature 설명을 먼저 만든다.
2. Slide 7-10에서 최종 방법론을 깔끔하게 정리한다.
3. Slide 12-15에서 "왜 최종 구조로 바뀌었는가"를 트러블슈팅 흐름으로 만든다.
4. Slide 16-17의 결과 슬라이드를 완성한다.
5. 마지막으로 Slide 2-4의 배경과 선행연구를 앞에 붙인다.

이 순서로 만들면 발표의 핵심 스토리인 "데이터와 최종 방법론 -> 실험 중 문제 진단 -> 최종 성과"가 자연스럽게 이어진다.

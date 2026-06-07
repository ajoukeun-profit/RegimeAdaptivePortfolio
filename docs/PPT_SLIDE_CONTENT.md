# PPT 슬라이드 콘텐츠

> 공통 포맷: `목차 -> 소제목 -> 관통 문장 -> 내용`  
> 현재 PPT 제작 흐름에 맞춰 총 13장 구성으로 정리한다.

## 최종 목차

| Section | Slide | 제목 | 역할 |
|---|---:|---|---|
| Background | 01 | 정적 포트폴리오의 한계 | 연구 필요성 |
| Background | 02 | 연구 목적 | 예측과 포트폴리오 연결 |
| Background | 03 | 선행 연구 | 기존 연구 대비 위치 |
| Data Structure | 04 | 데이터 구조 | 자산, feature, window, horizon |
| Data Structure | 05 | HMM Pseudo-label 생성 | 국면 label 생성 |
| Methodology | 06 | Conv1D+LSTM 국면 예측 모델 | Bear 확률 예측 |
| Methodology | 07 | 2-Regime MVO와 Weight Cap | 확률을 비중으로 변환 |
| Experiments | 08 | 실험 설계 및 평가 프레임워크 | 공통 세팅과 주요 비교 실험 |
| Experiments | 09 | 포트폴리오 모듈 선택 | Regime-MVO와 Weight Cap 선택 근거 |
| Experiments | 10 | 예측 모델 선택 | Label Ablation과 분류기 비교 |
| Results | 11 | 최종 모델 요약 | 예측 모델과 포트폴리오 모듈 연결 |
| Results | 12 | 최종 백테스트 설계와 결과 | 백테스트 방식과 성과 경로 |
| Results | 13 | 결과 해석 | 투자 baseline 대비 의미 해석 |

## Figure 활용 계획

| Figure | 파일 | 주 사용 슬라이드 | 용도 |
|---|---|---|---|
| Fig 01 | `outputs/figures/final/fig01_pipeline.png` | 02, 06, 07 | 전체 파이프라인, 모델-포트폴리오 연결 |
| Fig 02 | `outputs/figures/final/fig02_related_work.png` | 03 | 선행 연구 대비 본 프로젝트 위치 |
| Fig 03-A | `outputs/figures/final/fig03_static_dynamic_backtest.png` | 09, 12 | Regime-MVO와 비교군의 누적수익률/drawdown 경로. 09장에서는 선택 보조 자료 |
| Fig 03-B | `outputs/figures/final/fig03_main_result.png` | 12 | 최종 전략 성과 요약 |
| Fig 04 | `outputs/figures/final/fig04_classification_performance.png` | 10 | LR/RF baseline과 Conv1D+LSTM 분류 성능 비교. 10장에서는 선택 보조 자료 |
| Fig 05 | `outputs/figures/final/fig05_confusion_matrix.png` | 10 | Binary Soft Label confusion matrix. 공간 부족 시 생략 가능 |
| Fig 07 | `outputs/figures/final/fig07_ablation.png` | Backup | EW 1/N, 3-class MVO, binary soft MVO 비교 |
| Fig 09 | `outputs/figures/final/fig09_binary_mvo_weights.png` | 09 | cap별 Non-Bear/Bear MVO 비중 변화. 09장에서는 선택 보조 자료 |

---

# Background

## 01. 정적 포트폴리오의 한계

목차:

- Background

소제목:

- 왜 고정 비중 포트폴리오만으로는 부족한가?

관통 문장:

- 금융시장은 시간에 따라 위험과 상관관계가 변하기 때문에, 하나의 고정 비중이 모든 구간에서 최적이기는 어렵다.

내용:

| 구분 | 핵심 내용 |
|---|---|
| 정적 포트폴리오 | Buy & Hold, 60/40, EW 1/N |
| 기본 가정 | 과거 평균, 변동성, 공분산이 미래에도 비슷하게 유지됨 |
| 실제 시장 | 상승장, 하락장, 고변동성 구간에서 자산의 역할이 달라짐 |
| 문제의식 | 시장 국면이 바뀐다면 포트폴리오 비중도 바뀔 필요가 있음 |

하단 강조 문장:

- 정적 포트폴리오의 한계는 시장 국면 인식의 필요성으로 이어진다.

---

## 02. 연구 목적

목차:

- Background

소제목:

- 예측을 투자 의사결정으로 연결하기

관통 문장:

- 본 프로젝트의 목적은 국면을 맞히는 것에서 끝나는 것이 아니라, 예측 확률을 실제 포트폴리오 비중으로 변환하는 것이다.

내용:

| 단계 | 내용 |
|---|---|
| 1 | 시장 국면 예측 |
| 2 | Bear / Non-Bear 확률 산출 |
| 3 | 예측 확률을 MVO 비중에 반영 |
| 4 | 최종 포트폴리오 성과로 평가 |

평가 관점:

| 분류 성능 | 포트폴리오 성과 |
|---|---|
| Balanced Accuracy | 누적수익률 |
| Bear Recall | Sharpe |
| Macro F1 | MDD, Calmar |

사용 그림:

- `outputs/figures/final/fig01_pipeline.png`

하단 강조 문장:

- 분류 정확도는 중간 지표이고, 최종 평가는 포트폴리오 성과로 판단한다.

---

## 03. 선행 연구

목차:

- Background

소제목:

- 국면 추정, 딥러닝 예측, 포트폴리오 최적화의 결합

관통 문장:

- 본 프로젝트는 기존 연구의 세 흐름을 하나의 투자 파이프라인으로 연결한다.

내용:

| 흐름 | 기존 접근 | 본 프로젝트 |
|---|---|---|
| 국면 인식 | HMM으로 시장 상태 추정 | HMM label을 pseudo-label로 사용 |
| 딥러닝 예측 | 가격, 수익률, 국면 예측 | Conv1D+LSTM으로 Bear 확률 예측 |
| 포트폴리오 최적화 | 정적 MVO 또는 규칙 기반 전환 | 확률 가중 2-Regime MVO |

사용 그림:

- `outputs/figures/final/fig02_related_work.png`

하단 강조 문장:

- 핵심 차별점은 예측 모델을 포트폴리오 의사결정까지 연결했다는 점이다.

---

# Data Structure

## 04. 데이터 구조

목차:

- Data Structure

소제목:

- 4개 ETF의 30일 cross-asset sequence로 5일 후 국면 예측

관통 문장:

- 주식, 성장주, 금, 장기채의 상대 움직임을 이용해 시장 국면을 예측하는 supervised dataset을 구성한다.

내용:

| 항목 | 설정 |
|---|---|
| 사용 자산 | SPY, QQQ, GLD, TLT |
| 자산 역할 | 시장 대표, 성장주, 금, 장기채 |
| 입력 데이터 | daily OHLCV 기반 feature |
| 입력 길이 | 최근 30거래일 |
| 입력 형태 | 30 days x 40 features |
| 예측 horizon | 5거래일 후 |
| test 기간 | 2024-04-15 ~ 2026-05-15 |

자산 선택 이유:

| 자산 | 선택 이유 |
|---|---|
| SPY | 미국 주식시장 전체 흐름 대표 |
| QQQ | 성장주/기술주 성격 반영 |
| GLD | 위험회피 및 대체자산 후보 |
| TLT | 장기채, 방어 국면 반응 자산 |

하단 강조 문장:

- 국면 인식은 단일 자산 예측이 아니라 자산 간 관계 변화를 보는 문제다.

---

## 05. HMM Pseudo-label 생성

목차:

- Data Structure

소제목:

- 관측되지 않는 시장 국면을 HMM으로 추정

관통 문장:

- HMM은 시장의 잠재 상태를 추정하지만, 이 label은 실제 정답이 아니라 모델이 만든 pseudo-label이다.

내용:

| 단계 | 내용 |
|---|---|
| 1 | 시장 데이터로 HMM 학습 |
| 2 | 잠재 상태를 Bear / Neutral / Bull로 해석 |
| 3 | 각 상태의 posterior probability 산출 |
| 4 | supervised learning을 위한 pseudo-label로 사용 |

주의점:

| 구분 | 의미 |
|---|---|
| True label 아님 | 사람이 직접 부여한 정답이 아님 |
| Pseudo-label | HMM의 통계적 추정 결과 |
| Posterior 활용 | binary soft label 생성에 사용 |

하단 강조 문장:

- HMM label의 불확실성을 고려하기 위해 posterior probability를 활용한다.

---

# Methodology

## 06. Conv1D+LSTM 국면 예측 모델

목차:

- Methodology

소제목:

- 30일 시계열 패턴으로 Bear 확률 예측

관통 문장:

- Conv1D+LSTM은 최근 30일의 cross-asset 시계열 패턴을 이용해 미래 Bear 확률을 예측한다.

내용:

| 구성 | 역할 |
|---|---|
| 입력 | 30 days x 40 features |
| Conv1D | 짧은 구간의 local pattern 추출 |
| LSTM | 시간 순서와 누적 흐름 학습 |
| Classifier | P(Non-Bear), P(Bear) 출력 |

학습 설정:

| 항목 | 내용 |
|---|---|
| Target | Binary soft label |
| Loss | Soft cross entropy |
| Optimizer | AdamW |
| 안정화 | Dropout, early stopping, gradient clipping |

사용 그림:

- `outputs/figures/final/fig01_pipeline.png`

하단 강조 문장:

- 모델은 자산 비중을 직접 예측하지 않고, 포트폴리오 비중에 사용할 국면 확률을 예측한다.

---

## 07. 2-Regime MVO와 Weight Cap

목차:

- Methodology

소제목:

- 예측 확률을 포트폴리오 비중으로 변환

관통 문장:

- 예측된 Bear 확률을 이용해 Non-Bear MVO와 Bear MVO 비중을 가중 평균한다.

내용:

최종 비중:

```text
w_t =
P(Non-Bear) * w_NonBear
+ P(Bear) * w_Bear
```

MVO 제약:

```text
sum(w_i) = 1
0 <= w_i <= 0.4
```

선택 이유:

| 선택 | 이유 |
|---|---|
| 2-Regime MVO | Non-Bear와 Bear 구간의 평균-분산 구조를 따로 반영 |
| 확률 가중 | hard switching보다 부드러운 비중 조절 가능 |
| Weight cap 40% | 특정 자산 몰빵 완화 |

사용 그림:

- `outputs/figures/final/fig01_pipeline.png`

하단 강조 문장:

- 국면 예측 확률이 실제 포트폴리오 비중으로 연결되는 핵심 단계다.

---

# Experiments

## 08. 실험 설계 및 평가 프레임워크

목차:

- Experiments

소제목:

- 공통 실험 세팅과 주요 비교 실험 정리

관통 문장:

- 본 프로젝트는 단순히 모델 정확도를 보는 것이 아니라, 국면 예측이 실제 포트폴리오 성과로 이어지는지를 단계별 실험으로 검증한다.

내용:

공통 실험 세팅:

| 구분 | 내용 |
|---|---|
| 입력 데이터 | 30일 cross-asset sequence |
| 예측 대상 | 5거래일 후 시장 국면 |
| test 기간 | 2024-04-15 ~ 2026-05-15 |
| 최종 평가 | classification 성능 + portfolio backtest 성과 |

주요 비교 실험:

| 실험 | 목적 |
|---|---|
| 포트폴리오 모듈 선택 | Regime-MVO 효과와 Weight Cap 필요성 확인 |
| 예측 모델 선택 | Label Ablation과 LR/RF 대비 Conv1D+LSTM 성능 확인 |
| 최종 모델 요약 | 예측 모델과 포트폴리오 모듈을 하나의 전략으로 연결 |
| 최종 백테스트 설계와 결과 | 백테스트 방식과 성과 경로 확인 |
| 결과 해석 | 투자 baseline 대비 개선 지점 해석 |

평가 지표:

| 분류 성능 지표 | 포트폴리오 성과 지표 |
|---|---|
| Accuracy | Cumulative Return |
| Balanced Accuracy | Sharpe |
| Macro F1 | MDD |
| Bear Recall | Calmar |

하단 강조 문장:

- 중요한 실험을 먼저 보여주고, 실패 원인 진단은 뒤에서 최종 선택의 근거로 설명한다.

---

## 09. 포트폴리오 모듈 선택

목차:

- Experiments

소제목:

- 국면 확률을 어떤 포트폴리오 규칙으로 바꿀 것인가?

관통 문장:

- 최종 모델을 정하기 전에, 먼저 국면 정보가 포트폴리오에 실제로 도움이 되는지와 MVO 비중 쏠림을 어떻게 제어할지 확인한다.

내용:

검증 질문:

- 국면 정보는 포트폴리오 성과를 개선하고, weight cap은 MVO 비중 안정성을 개선하는가?

실험 flow에서의 위치:

```text
Portfolio module 선택 -> Prediction model 선택 -> Final model 요약 -> Final backtest
```

선택 과정:

| 단계 | 확인한 질문 | 비교군 | 결론 |
|---:|---|---|---|
| 1 | 국면 정보를 넣을 필요가 있는가? | 동일 cap 40% 조건: Regime-Agnostic MVO vs Regime-MVO | cap 조건을 통제한 상태에서 국면 확률 반영 효과를 확인 |
| 2 | MVO 비중 쏠림을 제어해야 하는가? | 동일 Regime-MVO 조건: no cap vs cap 50% vs cap 40% | 국면 구조를 고정한 상태에서 cap 효과를 확인 |
| 3 | 최종 포트폴리오 제약은 무엇인가? | cap 40%, cap 50% | 현재 test 구간에서는 cap 40%가 가장 균형적 |
| 최종 선택 | 예측 확률을 어떻게 비중으로 바꿀 것인가? | 2-Regime MVO + cap 40% | 최종 포트폴리오 모듈로 사용 |

Regime-MVO 필요성:

- 아래 비교는 둘 다 `cap 40%`를 적용한 상태이므로, 차이는 weight cap이 아니라 국면 정보를 반영했는지 여부로 해석한다.

| 전략 | 누적수익률 | Sharpe | MDD | Calmar | 해석 |
|---|---:|---:|---:|---:|---|
| Regime-Agnostic MVO cap 40% | 46.0% | 1.04 | -15.2% | 1.31 | 국면 구분 없음 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 | 국면 확률 반영 |

Weight cap 필요성:

- 아래 비교는 모두 `Binary Soft Regime-MVO` 구조를 고정하고, weight cap만 바꾼 실험이다.

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Binary Soft MVO no cap | 22.9% | 0.59 | -8.5% | 1.22 |
| Binary Soft MVO cap 50% | 47.9% | 1.37 | -8.3% | 2.48 |
| Binary Soft MVO cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

사용 그림:

- 선택 사항: `outputs/figures/final/fig03_static_dynamic_backtest.png`
- 선택 사항: `outputs/figures/final/fig09_binary_mvo_weights.png`
- 이 장은 그림보다 선택 과정 표를 중심으로 구성해도 충분함

하단 강조 문장:

- 포트폴리오 모듈은 2-Regime MVO + cap 40%로 확정하고, 다음 장에서는 이 모듈에 넣을 Bear 확률 예측 모델을 선택한다.

---

## 10. 예측 모델 선택

목차:

- Experiments

소제목:

- 어떤 label과 classifier가 Bear 확률을 가장 잘 만들 수 있는가?

관통 문장:

- 포트폴리오 모듈이 정해졌으므로, 이제 그 모듈에 입력할 Bear probability를 만들 최종 예측 모델을 선택한다.

내용:

실험 flow에서의 위치:

```text
Portfolio module 선택 -> Prediction model 선택 -> Final model 요약 -> Final backtest
```

선택 과정:

| 단계 | 확인한 질문 | 비교군 | 결론 |
|---:|---|---|---|
| 1 | 3-class가 적절한가? | 3-class hard | Neutral 구분이 불안정함 |
| 2 | 문제 정의를 단순화하면 나아지는가? | Binary hard | Bear / Non-Bear로 바꾸면 분류 성능이 안정화 |
| 3 | HMM label 불확실성을 반영하면 나아지는가? | Binary hard vs Binary soft | Binary soft가 Balanced Acc.와 Bear Recall 모두 가장 좋음 |
| 4 | 딥러닝 모델이 필요한가? | LR/RF vs Conv1D+LSTM | Conv1D+LSTM이 baseline보다 Bear 탐지에 유리 |
| 최종 선택 | Bear probability를 어떤 모델로 만들 것인가? | Binary Soft Conv1D+LSTM | 최종 예측 모델로 사용 |

Label ablation 결과:

| Label 방식 | Balanced Accuracy | Bear Recall | 결론 |
|---|---:|---:|---|
| 3-class hard | 51.9% | 60.5% | Neutral failure |
| Binary hard | 70.2% | 58.1% | 문제 정의 안정화 |
| Binary soft | 72.4% | 67.4% | HMM uncertainty 반영 |
| Binary soft + confidence | 72.4% | 67.4% | 추가 개선 없음 |

분류기 비교 결과:

| 모델 | Test Balanced Accuracy | Bear Recall | 해석 |
|---|---:|---:|---|
| Logistic Regression | 61.4% | 32.6% | 선형 baseline |
| Random Forest | 66.3% | 53.5% | 비선형 baseline |
| Binary Conv1D+LSTM | 70.2% | 58.1% | 시계열 모델 효과 |
| Binary Soft Conv1D+LSTM | 72.4% | 67.4% | 최종 선택 |

학습 안정성 해석:

| 점검 | 결론 |
|---|---|
| Gradient clipping, Dropout, Early stopping, AdamW | loss 폭주나 optimizer 문제가 핵심 원인은 아니었음 |
| Neutral class 지속 실패 | 학습 기법보다 label ambiguity가 핵심 병목으로 판단 |

사용 그림:

- 선택 사항: `outputs/figures/final/fig04_classification_performance.png`
- 선택 사항: `outputs/figures/final/fig05_confusion_matrix.png`
- 이 장은 그림보다 label/model 선택 과정 표를 중심으로 구성해도 충분함

하단 강조 문장:

- 예측 모델은 Binary Soft Conv1D+LSTM으로 확정하고, 다음 장에서는 포트폴리오 모듈과 결합한 최종 모델 구조를 정리한다.

---

# Results

## 11. 최종 모델 요약

목차:

- Results

소제목:

- 예측 모델과 포트폴리오 모듈을 하나의 전략으로 연결

관통 문장:

- 모델이 5거래일 후 하락 위험 확률을 예측하고, 그 확률에 따라 Non-Bear/Bear MVO 비중을 가중 평균한다.

내용:

최종 모델 / 전략:

```text
Conv1D+LSTM
(trained with Binary Soft Label)
-> Bear probability
-> 2-Regime MVO
-> Weight Cap 40%
```

| 구성 요소 | 최종 선택 |
|---|---|
| Label | Binary Soft Label |
| Classifier | Conv1D+LSTM |
| Portfolio | 2-Regime MVO |
| Constraint | Weight Cap 40% |
| Final Strategy | Binary Soft Regime-MVO cap 40% |

선택 근거 요약:

| 선택 | 근거 |
|---|---|
| Binary Soft Label | Neutral 구분이 불안정해 Bear / Non-Bear 중심으로 재정의 |
| Conv1D+LSTM | LR/RF baseline보다 Bear Recall과 Balanced Accuracy가 높음 |
| 2-Regime MVO | 동일 cap 조건에서 Regime-Agnostic MVO보다 성과와 위험 지표가 개선 |
| Weight Cap 40% | 동일 Regime-MVO 조건에서 no cap, cap 50% 대비 가장 균형적 |

다음 장 연결:

- 위 최종 모델을 test 구간에 적용해 투자 baseline과 백테스트 성과를 비교한다.

하단 강조 문장:

- 11장은 결과표가 아니라, 앞선 실험 선택을 하나의 최종 전략으로 정리하는 연결 장이다.

---

## 12. 최종 백테스트 설계와 결과

목차:

- Results

소제목:

- 최종 전략을 어떻게 test했는가?

관통 문장:

- test 구간에서 최종 전략을 투자 baseline과 같은 조건으로 비교해, 성과 경로와 위험 지표를 함께 확인한다.

내용:

Backtest setup:

| 항목 | 내용 |
|---|---|
| test 기간 | 2024-04-15 ~ 2026-05-15 |
| 최종 전략 | Binary Soft Regime-MVO cap 40% |
| 비중 산출 | Bear 확률로 Non-Bear/Bear MVO 비중을 가중 평균 |
| 비교군 | 60/40, Buy & Hold, EW 1/N, Regime-Agnostic MVO |
| 평가 지표 | Return, Sharpe, MDD, Calmar |

핵심 결과:

| 지표 | 최종 전략 | 비교 기준 |
|---|---:|---|
| Return | 53.7% | EW 1/N 50.9% |
| Sharpe | 1.48 | EW 1/N 1.41 |
| MDD | -9.0% | Buy & Hold -17.0% |
| Calmar | 2.55 | EW 1/N 2.47 |

사용 그림:

- `outputs/figures/final/fig03_static_dynamic_backtest.png`
- `outputs/figures/final/fig03_main_result.png`

하단 강조 문장:

- 12장은 “결과표 해석”보다 먼저, 동일 test 구간에서 최종 전략을 어떻게 검증했는지와 성과 경로를 보여주는 장이다.

---

## 13. 결과 해석

목차:

- Results

소제목:

- 어떤 baseline 대비 무엇이 개선되었는가?

관통 문장:

- 개선은 단순 수익률 극대화가 아니라, 하방 위험을 반영한 risk-return 균형 개선으로 해석한다.

내용:

성과 비교:

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| Regime-Agnostic MVO cap 40% | 46.0% | 1.04 | -15.2% | 1.31 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

해석:

| 비교 | 해석 |
|---|---|
| Buy & Hold 대비 | 누적수익률은 높고 MDD는 -17.0%에서 -9.0%로 낮음 |
| Regime-Agnostic MVO 대비 | 동일 cap 조건에서 국면 정보를 넣은 전략이 위험 지표 개선 |
| EW 1/N 대비 | 누적수익률과 Calmar는 높고 MDD는 유사 |

슬라이드 구성:

| 영역 | 내용 |
|---|---|
| Summary | 하방 위험을 반영한 risk-return 균형 개선 |
| 왼쪽 표 | 최종 전략과 투자 baseline 성과 비교 |
| 오른쪽 해석 | Buy & Hold, Regime-Agnostic MVO, EW 1/N 대비 개선 지점 |
| 하단 인사이트 | Downside risk, Regime information, Residual risk |

하단 강조 문장:

- 최종 전략은 수익률 극대화만이 아니라 하방 위험을 반영한 risk-return 균형 개선에 초점을 둔다.

---

## Backup. 연구 의의와 한계

목차:

- 질문 대비용

소제목:

- 위험 국면 확률을 포트폴리오 의사결정에 연결

관통 문장:

- 본 연구는 Bear 위험 확률을 포트폴리오 비중 조절에 연결해 risk-return 균형 개선 가능성을 보였지만, pseudo-label과 단일 test 구간이라는 한계가 있다.

내용:

연구 의의:

| 의의 | 내용 |
|---|---|
| 문제 정의 | 3-class Neutral failure 이후 Bear vs Non-Bear로 재정의해 투자 목적에 맞는 하방 위험 탐지에 집중 |
| 모델 연결 | Conv1D+LSTM의 Bear probability를 실제 포트폴리오 비중 산출에 연결 |
| 전략 검증 | 2-Regime MVO와 cap 40%로 동적 자산배분의 risk-return 균형 개선 가능성 확인 |

한계:

| 한계 | 내용 |
|---|---|
| HMM pseudo-label | 실제 정답 label이 아니라 HMM이 추정한 통계적 label |
| 단일 test 구간 | 다른 시장 국면에서도 walk-forward 검증 필요 |
| cap 40% 선택 | 현재 test 구간 기준이므로 validation 방식으로 선택하는 것이 더 엄밀함 |
| 현실 비용 미반영 | 거래비용, 세금, 시장충격을 충분히 반영하지 않음 |

하단 강조 문장:

- 핵심은 시장을 완벽히 예측하는 것이 아니라, 위험 국면 확률을 이용해 포트폴리오 비중을 더 합리적으로 조절하는 것이다.

---

# 질문 대비용 추가 실험

## A1. SPY/Cash Sanity Check

용도:

- 국면 확률 자체만으로 단순 위험 회피 전략을 만들었을 때 성과가 충분한지 확인한 보조 실험이다.

내용:

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Soft Binary SPY/Cash | 21.8% | 0.74 | -7.1% | 1.39 |
| Hard Binary SPY/Cash | 25.5% | 0.82 | -4.3% | 2.66 |

해석:

- drawdown은 낮지만 수익률이 낮아 최종 전략으로는 부족하다.
- 따라서 최종 스토리는 SPY/Cash threshold가 아니라 MVO 기반 전략으로 가져간다.

## A2. Oracle 실험 해석

용도:

- 분류기가 HMM pseudo-label을 100% 맞추면 포트폴리오 성과가 어디까지 좋아질 수 있는지 확인하는 상한선 실험이다.

내용:

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |
| Oracle Binary MVO cap 40% | 80.3% | 2.10 | -12.0% | 2.72 |

해석:

- Oracle은 실제 투자에서 쓸 수 있는 전략이 아니라 성능 상한선이다.
- 최종 모델과 Oracle 사이의 차이는 분류기 개선 여지가 아직 남아 있음을 의미한다.

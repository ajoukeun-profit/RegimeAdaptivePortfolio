# 15분 발표 목차: Market Regime-Aware Dynamic Portfolio Strategy

> 한국어 제목: 시장 국면 인식 기반 동적 포트폴리오 전략

> 금융 딥러닝 기초 기말 프로젝트
> 발표 목표: 모델 자체보다 **왜 시장 국면을 인식해야 하고, 그 인식이 포트폴리오 위험을 어떻게 줄였는지**를 명확하게 전달한다.

---

## 0. 발표 전체 메시지

**핵심 한 줄**

> 본 프로젝트는 HMM이 생성한 시장 국면 pseudo-label을 Conv1D+LSTM으로 예측하고, 그 예측 확률을 Regime-MVO 포트폴리오 비중에 연결하여 정적 포트폴리오 대비 하방 위험이 줄어드는지를 검증한 연구다.

**더 엄밀한 해석**

> HMM state는 실제 시장의 절대적 정답 label이 아니라 통계적 모델이 만든 pseudo-label이다. 따라서 HMM state 예측 정확도만으로 경제적 유효성을 주장할 수 없고, 예측된 state probability가 실제 포트폴리오 리밸런싱에서 MDD, Calmar, Sharpe와 같은 성과 지표를 개선하는지까지 확인해야 한다.

**발표에서 계속 반복할 키워드**

- 정적 포트폴리오 최적화의 한계
- 시장 국면의 비정상성
- HMM 기반 pseudo-label
- Conv1D+LSTM 기반 국면 예측
- 예측 확률을 활용한 Regime-MVO
- 분류 정확도보다 downstream portfolio performance
- 수익률 1등보다 MDD/Calmar 개선

**발표에서 피해야 할 표현**

- "HMM label이 진짜 정답이다"라고 말하지 않는다.
- "Regime-MVO가 정적 포트폴리오보다 수익률이 높다"라고 일반화하지 않는다.
- "Transformer를 고려하지 않았다"라고 말하지 않는다. Transformer 계열 구조도 검토했지만 최종 모델로 Conv1D+LSTM을 채택했다고 설명한다.

**발표에서 권장하는 표현**

- "HMM label은 시장 국면을 근사하기 위한 pseudo-label이다."
- "분류 모델은 최종 목적이 아니라 포트폴리오 의사결정을 위한 정보 생성 모듈이다."
- "본 전략의 경제적 기여는 누적수익률 극대화가 아니라 하방 위험 완화에 있다."

---

## 1. 전체 발표 구조

| Section | 한국어 | 시간 | 슬라이드 | 역할 |
|---|---|---:|---:|---|
| 1. Background | 연구 배경 | 2분 | 1~2 | 왜 이 문제가 중요한지 설명 |
| 2. Related Work | 선행연구 | 2분 | 3~4 | 기존 연구 흐름과 연구 공백 제시 |
| 3. Methodology | 방법론 | 4분 | 5~7 | HMM 라벨링, Conv1D+LSTM, Regime-MVO 구조 설명 |
| 4. Experiments | 실험 설계 | 2분 | 8 | 분류 성능과 백테스트 평가 기준 설명 |
| 5. Results | 결과 분석 | 4분 | 9~11 | 최종 백테스트, Ablation, 2022 검증 |
| 6. Conclusion | 한계와 결론 | 1분 | 12 | 한계와 향후 개선 방향 정리 |
| **Total** | **합계** | **15분** | **12장** |  |

**발표 목차 표기 추천**

1. **Background**: 연구 배경
2. **Related Work**: 선행연구
3. **Methodology**: 방법론
4. **Experiments**: 실험 설계
5. **Results**: 결과 분석
6. **Conclusion**: 한계와 결론

---

## 2. 목차별 포함 내용

### 1. Background: 연구 배경

**넣을 내용**

- 전통적 포트폴리오 최적화는 기대수익률과 공분산을 과거 데이터로부터 추정하고, 그 추정값이 테스트 기간에도 어느 정도 유지된다고 가정한다.
- 하지만 실제 금융시장은 상승장, 하락장, 중립장처럼 국면이 바뀐다.
- 국면이 바뀌면 자산의 수익률, 변동성, 상관관계도 달라진다.
- 따라서 하나의 고정 비중을 유지하는 정적 포트폴리오만으로는 국면 변화에 따른 위험 구조 변화를 반영하기 어렵다.
- 본 프로젝트는 "시장 국면을 예측하면 포트폴리오의 하방 위험을 줄일 수 있는가?"라는 질문에서 출발한다.

**핵심 문장**

> 금융시장은 하나의 고정된 분포를 따르지 않기 때문에, 포트폴리오 전략도 시장 국면 변화에 대응할 필요가 있다.

**수식으로 짚을 포인트**

전통적 MVO는 다음과 같은 평균-분산 trade-off를 최적화한다.

$$
\max_w \left( w^\top \mu - \lambda w^\top \Sigma w \right)
$$

여기서 `μ`와 `Σ`는 실제 미래 값을 아는 것이 아니라 과거 데이터에서 추정한 값이다. 국면이 바뀌면 이 추정값의 안정성이 약해질 수 있다.

**넣을 Figure**

- 기본적으로 텍스트 중심 슬라이드로 처리
- 필요하면 [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png)를 맛보기 결과로 일부 사용

---

### 2. Related Work: 선행연구

**넣을 내용**

- Markowitz: MVO의 기본 이론
- Hamilton: regime switching model
- HMM 기반 시장 국면 탐지 연구
- Regime-switching asset allocation 연구
- 딥러닝 금융 시계열 예측 연구
- 기존 연구의 한계: 국면을 쓰더라도 룰 기반이거나, 예측 확률을 포트폴리오 비중에 직접 연결하는 구조가 제한적임

**핵심 문장**

> 본 프로젝트는 HMM 기반 국면 정의, Conv1D+LSTM 기반 국면 예측, Regime-MVO 기반 자산배분을 하나의 파이프라인으로 연결한다.

**넣을 Figure**

- [fig02_related_work.png](../outputs/figures/final/fig02_related_work.png)

---

### 3. Methodology: 방법론

**넣을 내용**

- 데이터: SPY, QQQ, GLD, TLT
- HMM 라벨링:
  - 504일 rolling window
  - 3-state Gaussian HMM
  - Sharpe 기준 Bear/Neutral/Bull 매핑
  - 이 label은 true label이 아니라 pseudo-label임을 명시
- 데이터셋:
  - 입력: 30일 x 40피처
  - 타겟: 5일 후 SPY 국면
- 모델:
  - Conv1D: 단기 패턴 추출
  - LSTM: 시계열 흐름 학습
  - 출력: Bear/Neutral/Bull 확률
- 포트폴리오:
  - 국면별 MVO 비중 계산
  - 예측 확률로 비중 가중 평균

**핵심 수식**

$$
w_t
=
P(\text{Bear}) w_{\text{Bear}}
+
P(\text{Neutral}) w_{\text{Neutral}}
+
P(\text{Bull}) w_{\text{Bull}}
$$

**엄밀한 설명 포인트**

HMM label 생성은 다음과 같이 볼 수 있다.

$$
\hat{y}^{HMM}_t = g_{\phi}(x_{1:t})
$$

딥러닝 모델은 실제 시장의 절대적 정답이 아니라 이 pseudo-label을 근사한다.

$$
f_{\theta}(X_t) \approx \hat{y}^{HMM}_{t+h}
$$

따라서 분류 성능은 중간 지표이고, 최종 평가는 포트폴리오 백테스트에서 이루어진다.

**넣을 Figure**

- [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png)

---

### 4. Experiments: 실험 설계

**넣을 내용**

- 실험 목적:
  - 모델이 HMM pseudo-label을 어느 정도 예측할 수 있는가?
  - 그 예측 확률이 실제 포트폴리오 리밸런싱에서 경제적 의미를 갖는가?
- 분류 평가 지표:
  - Accuracy
  - Bear Recall
  - Neutral Recall
  - Bull Recall
- 포트폴리오 평가 지표:
  - Cumulative Return
  - Sharpe Ratio
  - MDD
  - Calmar Ratio
- 비교 전략:
  - Buy & Hold
  - 60/40
  - EW 1/N
  - Regime-Agnostic MVO
  - DL Regime SPY/Cash
  - Regime-MVO
  - Oracle (HMM labels)
- Test period: 2024.04 ~ 2026.05
- 추가 검증: 2022 하락장 스트레스 테스트

**핵심 문장**

> HMM label은 pseudo-label이므로 분류 정확도만으로 결론을 내리지 않고, 예측 확률이 포트폴리오 MDD와 Calmar 개선으로 이어지는지를 함께 평가한다.

**평가 논리**

$$
\text{High Classification Accuracy}
\nRightarrow
\text{High Economic Profit}
$$

따라서 발표에서는 "분류 성능"과 "투자 성과"를 분리해서 보여주되, 둘을 하나의 파이프라인으로 연결해 해석한다.

**넣을 Figure**

- [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png)
- [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png)

---

### 5. Results: 결과 분석

**넣을 내용**

- 분류 성능 결과:
  - 최종 Accuracy 61.9%
  - Bear Recall 60.5%
  - Bull Recall 95.1%
  - Neutral Recall 0%는 한계
- 최종 백테스트 결과:
  - Regime-MVO 누적수익률 35.3%
  - Sharpe 1.10
  - MDD -7.2%
  - Calmar 2.16
- 정적 benchmark 대비 해석:
  - Regime-MVO는 누적수익률 1등 전략이 아님
  - Buy & Hold, EW 1/N, Regime-Agnostic MVO보다 누적수익률은 낮음
  - 대신 MDD가 -7.2%로 가장 작고 하방 위험 관리가 우수함
- 핵심 비교:
  - Regime-Agnostic MVO: MDD -20.8%
  - Regime-MVO: MDD -7.2%
  - 국면 conditioning으로 MDD 13.6%p 개선
- Ablation:
  - DL Regime SPY/Cash: MDD -7.4%
  - Regime-MVO: MDD -7.2%
  - Oracle (HMM labels): MDD -6.2%
  - 현재 분류기도 MDD 관리 측면에서는 실용적
- 2022 하락장 분석:
  - DL Regime SPY/Cash: MDD -10.5%
  - Regime-MVO: MDD -22.2%
  - 2022년은 주식과 장기채가 동반 하락한 금리인상형 Bear
  - TLT 중심 방어 자산 설계의 한계

**핵심 문장**

> Regime-MVO는 수익률 1등 전략이 아니라, 국면 인식을 통해 MDD와 Calmar를 개선하는 위험관리 전략이다.

**주의해서 말할 점**

> "정적 포트폴리오보다 수익률이 높았다"가 아니라, "정적 포트폴리오 및 국면 무시 MVO 대비 drawdown을 완화했다"라고 말한다.

**넣을 Figure**

- [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png)
- [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png)
- [fig06_regime_conditional.png](../outputs/figures/final/fig06_regime_conditional.png)
- [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png)
- [fig08_2022_bear.png](../outputs/figures/final/fig08_2022_bear.png)

---

### 6. Conclusion: 한계와 결론

**넣을 내용**

- 결론:
  - HMM으로 시장 국면을 pseudo-label 형태로 정의했다.
  - Conv1D+LSTM은 Bear 국면 탐지력을 개선했다.
  - 예측된 국면 확률을 Regime-MVO에 연결했을 때 MDD가 크게 줄었다.
  - 단, 이 결과는 수익률 극대화가 아니라 위험관리 성과로 해석해야 한다.
- 한계:
  - Neutral Recall 0%
  - 전체 supervised sample은 약 698개이며, 실제 train sample은 488개로 작음
  - 2022년 금리인상형 Bear에서 TLT 방어 실패
- 향후 개선:
  - 현금, 단기채, 금 등 방어 자산 확대
  - Bear 국면 세분화
  - MDD/Calmar를 직접 최적화하는 목적함수 도입
  - 더 긴 기간의 데이터 또는 더 촘촘한 샘플링으로 학습 데이터 확장

**마지막 문장**

> 이 프로젝트의 핵심은 시장 가격을 정확히 맞히는 것이 아니라, 위험한 국면을 인식하고 그에 맞게 포트폴리오를 조정하는 것이다.

**넣을 Figure**

- 보통 결론 슬라이드는 Figure 없이 3줄 요약으로 처리

---

## 3. 슬라이드별 구성안

### 1. Title

**제목**

시장 국면 인식 기반 동적 포트폴리오 전략

**말할 내용**

- 본 프로젝트는 가격을 직접 예측하기보다 시장 국면을 예측한다.
- 예측된 국면 확률을 포트폴리오 비중 조절에 사용한다.

**키워드**

`Market Regime`, `Dynamic Asset Allocation`, `Conv1D+LSTM`, `Regime-MVO`

**Figure**

- 사용하지 않음

---

### 2. 연구 배경: 왜 정적 포트폴리오가 부족한가

**핵심 질문**

> 시장 상황이 계속 바뀌는데, 하나의 고정된 포트폴리오 비중이 충분할까?

**세부 내용**

- 전통적 MVO는 평균 수익률과 공분산을 추정해 최적 비중을 계산한다.
- 하지만 금융시장은 비정상적이다.
- 상승장, 중립장, 하락장에서는 수익률 분포와 자산 간 상관관계가 달라진다.
- 따라서 포트폴리오도 시장 국면에 따라 달라져야 한다.

**키워드**

`Mean-Variance Optimization`, `Non-stationarity`, `Market Regime`, `Risk Management`

**Figure**

- 선택 사항: 간단한 텍스트 슬라이드로 처리
- 만약 결과를 먼저 보여주고 싶으면 [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png)를 일부 사용해도 됨

---

### 3. 선행연구: MVO, Regime Switching, HMM

**세부 내용**

- Markowitz(1952): 평균-분산 최적화의 출발점
- Hamilton(1989): 경제/금융 시계열을 국면 전환 구조로 해석
- Ang & Bekaert(2002): 국면 변화가 자산배분에 미치는 영향 분석
- HMM 기반 동적 자산배분 연구: 숨겨진 시장 국면을 추정해 포트폴리오에 반영
- Costa & Kwon(2020): 국면별 기대수익률/공분산을 MVO 입력으로 사용하는 Regime-Switching MVO

**우리 프로젝트의 위치**

> 기존 연구가 "국면을 반영한 자산배분"을 다뤘다면, 본 프로젝트는 HMM으로 만든 국면 라벨을 Conv1D+LSTM으로 예측하고, 그 예측 확률을 MVO 비중에 연결했다.

**키워드**

`Markowitz`, `Hamilton`, `HMM`, `Regime Switching`, `Regime-dependent MVO`

**Figure**

- [fig02_related_work.png](../outputs/figures/final/fig02_related_work.png)

---

### 4. 연구 질문과 기여

**연구 질문**

> HMM으로 정의한 시장 국면 pseudo-label을 예측하고, 그 예측 확률을 포트폴리오 비중에 연결하면 하방 위험을 줄일 수 있는가?

**세부 질문**

- HMM으로 생성한 Bear/Neutral/Bull pseudo-label을 딥러닝 모델이 예측할 수 있는가?
- 이 pseudo-label 예측이 단순한 통계적 분류에 그치지 않고 포트폴리오 성과로 이어지는가?
- 예측된 국면 확률을 사용하면 국면을 무시한 MVO보다 MDD를 줄일 수 있는가?
- 2022년처럼 주식과 채권이 동시에 하락한 구간에서도 전략이 잘 작동하는가?

**기여**

- HMM 라벨링과 딥러닝 분류기를 결합
- 국면 확률을 hard switching이 아니라 MVO 비중의 확률 가중 평균으로 사용
- 단순 누적수익률이 아니라 MDD, Calmar, 2022 하락장 검증까지 포함
- HMM label의 pseudo-label 성격을 인정하고, 최종 경제적 검증을 백테스트로 수행

**키워드**

`Regime Probability`, `Soft Allocation`, `MDD`, `Calmar`, `2022 Bear Market`

**Figure**

- 사용하지 않음

---

### 5. 전체 파이프라인

**세부 내용**

1. SPY/QQQ/GLD/TLT 일별 OHLCV 수집
2. SPY 기준 HMM으로 Bear/Neutral/Bull 라벨 생성
3. 4개 자산의 피처를 결합해 30일 x 40피처 입력 데이터 생성
4. Conv1D+LSTM으로 5일 후 시장 국면 확률 예측
5. 예측 확률로 국면별 MVO 비중을 가중 평균
6. 2024.04~2026.05 구간에서 백테스트

**키워드**

`OHLCV`, `HMM Labeling`, `30 days x 40 features`, `5-day ahead`, `Regime-MVO`

**Figure**

- [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png)

---

### 6. 데이터와 HMM 라벨링

**데이터**

- 자산: SPY, QQQ, GLD, TLT
- 피처: 각 자산별 OHLCV 기반 피처 및 기술지표
- 최종 supervised sample: 총 698개 내외
- Train/Valid/Test: 488 / 105 / 105
- 최종 입력 shape: `(sample, 30, 40)`
- 타겟: 5일 후 SPY 기준 시장 국면

**HMM 라벨링**

- 504일 rolling window
- 3-state Gaussian HMM
- 각 hidden state를 Sharpe 기준으로 Bear/Neutral/Bull에 매핑
- HMM label은 관측된 진짜 정답이 아니라 모델 기반 pseudo-label

**말할 포인트**

- 국면 라벨은 사람이 임의로 붙인 것은 아니지만, 그렇다고 절대적 정답도 아니다.
- HMM이 추정한 hidden state를 Sharpe 특성에 따라 Bear/Neutral/Bull로 이름 붙인 pseudo-label이다.
- SPY를 기준으로 시장 국면을 정의하고, QQQ/GLD/TLT는 국면 예측을 돕는 cross-asset 정보로 사용했다.
- 따라서 모델의 분류 정확도는 "HMM이 만든 국면 체계를 얼마나 잘 근사했는가"를 의미한다.
- 최종적으로 중요한 것은 이 국면 확률이 포트폴리오 위험 조절에 유용했는지이다.

**수식**

$$
\hat{y}^{HMM}_t = g_{\phi}(x_{1:t})
$$

$$
f_{\theta}(X_t) \approx \hat{y}^{HMM}_{t+5}
$$

**키워드**

`Rolling Window`, `Gaussian HMM`, `Hidden State`, `Pseudo-label`, `Sharpe Mapping`, `Cross-asset Features`

**Figure**

- 필요하면 [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png)의 왼쪽 절반을 재사용

---

### 7. 모델과 포트폴리오 구성

**Conv1D+LSTM 모델**

- 입력: 과거 30일 x 40개 피처
- Conv1D: 단기 국소 패턴 추출
- LSTM: 30일 시계열 흐름 학습
- 출력: `[P(Bear), P(Neutral), P(Bull)]`
- 손실함수: class-weighted cross entropy
- Optimizer: AdamW
- 최종 checkpoint 선택 기준: validation balanced accuracy

**Regime-MVO**

훈련 데이터에서 국면별 Sharpe 최대화 비중을 계산한다.

| 국면 | MVO 비중 | 해석 |
|---|---|---|
| Bear | TLT 100% | 훈련 구간에서 채권이 방어 자산 역할 |
| Neutral | SPY 51% + GLD 49% | 중립 구간에서 주식/금 균형 |
| Bull | SPY 95% + QQQ 5% | 상승 구간에서 주식 중심 |

최종 비중:

$$
w_t
=
P(\text{Bear}) w_{\text{Bear}}
+
P(\text{Neutral}) w_{\text{Neutral}}
+
P(\text{Bull}) w_{\text{Bull}}
$$

**말할 포인트**

- 국면을 딱 하나로 선택하는 hard switching이 아니다.
- 모델이 예측한 확률만큼 각 국면 포트폴리오를 부드럽게 섞는다.
- 이 방식은 모델의 불확실성을 포트폴리오 비중에 직접 반영한다.
- MVO의 기대수익률과 공분산은 미래를 아는 값이 아니라 훈련 구간 과거 수익률에서 추정한 값이다.

**모델 구조 선택 관련 보충**

- Transformer 계열 구조도 검토되었다.
- 하지만 금융 데이터는 noise가 크고 학습 표본이 작아 self-attention의 높은 표현력이 과적합으로 이어질 수 있다.
- Conv1D+LSTM은 국소 패턴 추출과 순차적 정보 누적이라는 inductive bias가 있어 짧은 금융 시계열에 더 보수적이고 안정적인 선택이다.
- 따라서 Transformer가 열등해서 제외한 것이 아니라, 본 데이터 조건에서는 Conv1D+LSTM이 최종 파이프라인에 더 적합하다고 판단했다.

**키워드**

`Conv1D`, `LSTM`, `Weighted Cross Entropy`, `AdamW`, `Softmax`, `Sharpe Maximization`, `Probability-weighted MVO`

**Figure**

- [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png)

---

### 8. 분류 성능: Bear 탐지 개선

**핵심 결과**

| 실험 | Accuracy | Bear Recall | Neutral Recall | Bull Recall |
|---|---:|---:|---:|---:|
| Phase 1 Baseline | 57.1% | 34.9% | 23.8% | 97.6% |
| Phase 1 Augmentation | 61.0% | 46.5% | 33.3% | 90.2% |
| Phase 2 Multi-asset label | 59.8% | 58.8% | 25.3% | 80.6% |
| Phase 3 Final | 61.9% | 60.5% | 0.0% | 95.1% |

**해석**

- Accuracy만 보면 61.9%로 아주 높아 보이지 않을 수 있다.
- 그러나 이 프로젝트의 목적은 가격 방향을 완벽히 맞히는 것이 아니라 위험 국면을 인식하는 것이다.
- 포트폴리오 관점에서는 Bear Recall 개선이 특히 중요하다.
- Bear Recall이 34.9%에서 60.5%로 상승하면서 위험 구간 탐지력이 개선됐다.
- Neutral Recall 0%는 한계로 인정해야 한다. 중립 국면은 경계가 모호하고 라벨 자체가 구조적으로 어렵다.
- HMM label이 pseudo-label이라는 점 때문에, classification metric은 경제적 성과의 충분조건이 아니다.
- 따라서 다음 슬라이드에서 백테스트로 downstream usefulness를 검증한다.

**키워드**

`Accuracy`, `Bear Recall`, `Pseudo-label`, `Class Imbalance`, `Neutral Ambiguity`, `Downstream Validation`

**Figure**

- [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png)
- [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png)

---

### 9. 최종 백테스트: 수익률보다 위험관리

**Test Period**

2024.04 ~ 2026.05

**핵심 결과**

| 전략 | 누적수익 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| Regime-Agnostic MVO | 64.8% | 1.11 | -20.8% | 1.30 |
| DL Regime SPY/Cash | 21.9% | 0.73 | -7.4% | 1.35 |
| Regime-MVO | 35.3% | 1.10 | -7.2% | 2.16 |
| Oracle (HMM labels) | 41.6% | 1.16 | -6.2% | 2.91 |

**해석**

- Regime-MVO는 누적수익률 1등 전략이 아니다.
- 특히 Buy & Hold, EW 1/N, Regime-Agnostic MVO보다 누적수익률은 낮다.
- 대신 MDD가 -7.2%로 낮고 Calmar가 2.16으로 개선됐다.
- 국면을 무시한 MVO는 누적수익률은 높지만 MDD가 -20.8%까지 악화됐다.
- 즉, MVO 자체보다 **국면 conditioning**이 위험관리의 핵심이다.
- 따라서 본 결과는 "수익률 우위"가 아니라 "하방 위험 완화"로 해석해야 한다.

**정적 포트폴리오의 의미**

- Buy & Hold: SPY 100% 고정
- EW 1/N: SPY/QQQ/GLD/TLT 각 25% 고정
- 60/40: 사전에 정한 주식/현금 또는 주식/채권 비중 고정
- Regime-Agnostic MVO: 훈련 데이터 전체로 한 번 계산한 MVO 비중을 테스트 기간에 고정
- Regime-MVO: 매 시점 예측 확률에 따라 비중을 바꾸는 동적 포트폴리오

**키워드**

`Backtest`, `Static Portfolio`, `Dynamic Portfolio`, `MDD`, `Calmar`, `Regime Conditioning`, `Risk-adjusted Return`

**Figure**

- [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png)
- [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png)

---

### 10. 국면별 성과와 Ablation

**국면별 성과**

- Bear 구간에서 Regime-MVO가 가장 중요한 역할을 한다.
- Buy & Hold는 Bear 구간에서 큰 손실을 보지만, Regime-MVO는 하락 위험을 줄인다.
- 이 결과는 "국면을 예측하는 이유"를 가장 직접적으로 보여준다.

**Ablation 핵심**

| 구성 | MDD | Calmar | 의미 |
|---|---:|---:|---|
| Buy & Hold | -17.0% | 1.26 | 기본 비교군 |
| EW 1/N | -8.8% | 2.47 | 단순하지만 강한 정적 분산 benchmark |
| Regime-Agnostic MVO | -20.8% | 1.30 | 국면을 무시하면 MVO가 위험해질 수 있음 |
| DL Regime SPY/Cash | -7.4% | 1.35 | 국면 신호만으로도 MDD 개선 |
| Regime-MVO | -7.2% | 2.16 | 국면 신호와 MVO 결합 |
| Oracle (HMM labels) | -6.2% | 2.91 | HMM pseudo-label을 미리 아는 상한 |

**해석**

- Regime-MVO는 Oracle (HMM labels)과 MDD 차이가 약 1.0%p다.
- 이는 현재 분류기가 MDD 관리 측면에서는 꽤 실용적인 수준임을 의미한다.
- 다만 Calmar 차이는 남아 있어 수익률/리밸런싱 측면의 개선 여지는 있다.

**키워드**

`Regime-conditional Performance`, `Ablation Study`, `Oracle HMM Labels`, `Practical Classifier`

**Figure**

- [fig06_regime_conditional.png](../outputs/figures/final/fig06_regime_conditional.png)
- [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png)

---

### 11. 2022 하락장 검증: 한계까지 보여주기

**왜 2022년을 따로 보는가**

- 2022년은 Fed 금리 인상으로 주식과 장기채가 동시에 하락한 구간이다.
- 일반적인 주식-채권 분산 효과가 약하게 작동했다.
- 따라서 방어 자산 설계가 정말 잘 되었는지 확인하기 좋은 스트레스 테스트다.

**결과**

| 전략 | MDD |
|---|---:|
| Buy & Hold | -20.5% |
| 60/40 | -12.5% |
| EW 1/N | -21.7% |
| DL Regime SPY/Cash | -10.5% |
| Regime-MVO | -22.2% |
| Regime Momentum Tilt | -21.3% |

**해석**

- 2022년에는 DL Regime SPY/Cash가 가장 잘 방어했다.
- Regime-MVO는 Bear 국면 비중이 TLT 100%였기 때문에 금리인상형 Bear에 취약했다.
- 이 결과는 실패가 아니라 중요한 한계 분석이다.
- 향후에는 TLT만 방어 자산으로 두지 않고, 현금/단기채/금 등을 함께 고려해야 한다.

**키워드**

`2022 Bear Market`, `Rate-hike Regime`, `Stock-Bond Correlation`, `Cash`, `Defensive Asset`

**Figure**

- [fig08_2022_bear.png](../outputs/figures/final/fig08_2022_bear.png)

---

### 12. 결론과 향후 개선

**결론 3줄**

1. HMM state는 절대적 정답이 아니라 pseudo-label이므로, 분류 정확도만으로 경제적 유효성을 주장할 수 없다.
2. Conv1D+LSTM 분류기는 Bear 탐지력을 개선했고, 이 국면 확률은 Regime-MVO의 동적 비중 조절에 사용됐다.
3. Regime-MVO는 수익률 1등 전략은 아니지만, 정적 포트폴리오 및 국면 무시 MVO 대비 drawdown을 완화하는 위험관리 전략으로 의미가 있다.

**한계**

- Neutral Recall이 0%로 중립 국면 식별이 약하다.
- 전체 supervised sample은 약 698개이고 train sample은 488개로, 딥러닝 모델 학습에는 작다.
- HMM label 자체가 pseudo-label이므로 label quality에 모델 성능이 제한된다.
- 2022년처럼 주식과 채권이 동시에 하락하는 금리인상형 Bear에서는 TLT 중심 방어가 실패할 수 있다.

**향후 개선**

- 현금, 단기채, 금 등 방어 자산 후보 확대
- Bear 국면 내에서도 금리인상형/경기침체형 Bear를 세분화
- MDD 또는 Calmar를 직접 반영하는 포트폴리오 목적함수 도입
- 일별 sliding window 또는 더 긴 데이터로 학습 샘플 확대
- HMM label 생성 방식 개선 또는 regime definition 자체 재검토
- Transformer/Temporal Fusion Transformer 등 attention 계열 모델은 더 큰 데이터셋에서 재검토 가능

**마지막 멘트**

> 이 프로젝트의 핵심은 가격을 정확히 맞히는 것이 아니라, 통계적으로 정의한 위험 국면 신호가 실제 포트폴리오 위험 조절에 도움이 되는지를 검증하는 것이다.

**Figure**

- 사용하지 않음

---

## 4. Figure 사용 계획 요약 및 검증

본 연구의 최종 질문은 "HMM pseudo-label을 예측한 결과가 정적 포트폴리오 대비 실제 리밸런싱 성과, 특히 하방 위험 관리에 도움이 되었는가?"이다. 따라서 발표용 figure는 분류 성능보다 **정적 benchmark vs 동적 Regime-MVO 백테스트 비교**를 중심에 둔다. 기존 figure의 강조점이 섞여 있던 문제를 보완하기 위해, 정적/동적 백테스트 경로를 직접 비교하는 `fig03_static_dynamic_backtest.png`를 추가했고, `fig03_main_result.png`는 핵심 성과 지표 요약 그림으로 재정의했다.

### 4.1 현재 figure 적합성 평가

| Figure | 파일 | 현재 역할 | 평가 | 발표 사용 방식 |
|---|---|---|---|---|
| Fig 01 | [fig01_pipeline.png](../outputs/figures/final/fig01_pipeline.png) | 전체 파이프라인 | 사용 적합. 단, HMM label이 pseudo-label이라는 점을 말로 보완해야 함 | Methodology 첫 장 |
| Fig 02 | [fig02_related_work.png](../outputs/figures/final/fig02_related_work.png) | 선행연구 비교 | 선택 사항. 15분 발표에서는 시간이 부족하면 생략 가능 | Related Work 보조 |
| Fig 03-A | [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png) | 정적 benchmark vs 동적 Regime-MVO 백테스트 경로 | 새로 추가. 누적수익률과 drawdown path를 함께 보여주는 Results 메인 figure | Results 메인 |
| Fig 03-B | [fig03_main_result.png](../outputs/figures/final/fig03_main_result.png) | 누적수익률/MDD/Calmar 요약 | 수정 완료. Regime-MVO가 수익률 1등이 아니라 MDD 완화 전략임을 보여줌 | Results 보조 |
| Fig 04 | [fig04_classification_performance.png](../outputs/figures/final/fig04_classification_performance.png) | 분류 성능 변화 | 보조 지표로 적합. 단, 발표 초점이 분류 대회처럼 보이지 않게 주의 | Experiments 보조 |
| Fig 05 | [fig05_confusion_matrix.png](../outputs/figures/final/fig05_confusion_matrix.png) | 최종 confusion matrix | 한계 설명에 유용. Neutral Recall 0%를 숨기지 않는 장점 | Fig 04와 한 장에 묶거나 appendix |
| Fig 06 | [fig06_regime_conditional.png](../outputs/figures/final/fig06_regime_conditional.png) | HMM pseudo-label 국면별 성과 | 수정 완료. Bear 구간 방어 논리를 보여주되, 전체 국면에서 항상 우위라고 과장하지 않음 | Bear 방어 근거 및 국면별 진단 |
| Fig 07 | [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png) | 구성요소 기여 및 benchmark 점검 | 수정 완료. EW 1/N을 포함했고 Oracle 표현을 HMM pseudo-label 상한으로 통일함 | Results 보조 또는 appendix |
| Fig 08 | [fig08_2022_bear.png](../outputs/figures/final/fig08_2022_bear.png) | 2022 스트레스 테스트 | 매우 중요. 다만 성공 사례가 아니라 한계 분석으로 사용해야 함 | Limitation 메인 |

### 4.2 수정 전 figure 설계의 핵심 문제와 수정 결과

1. **정적 vs 동적 백테스트 비교가 한 장으로 명확히 보이지 않는다.**
   - 연구 목적상 가장 중요한 비교는 정적 포트폴리오와 동적 Regime-MVO이다.
   - 기존 Fig 03은 MDD/Calmar 막대로 이 비교를 보여주지만, 시간축 누적수익률과 drawdown curve가 없었다.
   - 수정 결과: [fig03_static_dynamic_backtest.png](../outputs/figures/final/fig03_static_dynamic_backtest.png)를 추가해 누적수익률 경로와 drawdown 경로를 함께 보여준다.

2. **Fig 07 ablation에서 EW 1/N이 빠져 있다.**
   - EW 1/N은 테스트 구간에서 Sharpe와 Calmar가 매우 좋은 정적 benchmark이다.
   - 이를 ablation에서 제외하면, 좋은 benchmark를 의도적으로 뺀 것처럼 보일 수 있다.
   - 수정 결과: [fig07_ablation.png](../outputs/figures/final/fig07_ablation.png)에 EW 1/N을 포함했다.

3. **Oracle 표현이 엄밀하지 않다.**
   - 과거 일부 figure/표에서는 Oracle을 실제 정답처럼 표현했지만, 여기서 label은 실제 시장의 절대적 정답이 아니라 HMM pseudo-label의 test label이다.
   - 더 엄밀한 표현은 `Oracle (HMM labels)` 또는 `Oracle (Pseudo-label upper bound)`이다.
   - 수정 결과: 새로 생성한 Fig 03-A와 수정한 Fig 07에서는 `Oracle (HMM labels)`로 통일했다.

4. **분류 성능 figure가 앞에 강하게 나오면 연구 목적이 흐려질 수 있다.**
   - HMM label은 pseudo-label이므로, classification accuracy가 최종 목적처럼 보이면 안 된다.
   - Fig 04, Fig 05는 "포트폴리오 성과를 설명하기 위한 중간 진단"으로 배치해야 한다.

5. **2022 figure는 성공 사례가 아니라 한계 사례다.**
   - Fig 08에서 Regime-MVO는 2022년에 좋지 않다.
   - 따라서 이를 "하락장에서도 Regime-MVO가 좋았다"가 아니라 "금리인상형 Bear에서는 TLT 중심 방어가 실패했다"는 한계 분석으로 사용해야 한다.

### 4.3 새로 추가한 핵심 figure

**Fig 03-A. Static vs Dynamic Backtest Curve**

목적:

- 정적 포트폴리오와 동적 Regime-MVO의 시간축 성과를 직접 비교한다.

포함 전략:

- Buy & Hold
- EW 1/N
- 60/40
- Regime-Agnostic MVO
- Regime-MVO
- Oracle (HMM labels)

패널 구성:

- 위: cumulative return curve
- 아래: drawdown curve

이 figure가 있어야 "정적/동적 백테스팅을 비교했다"는 메시지가 가장 명확해진다.

### 4.4 추가로 있으면 더 좋은 figure

**추가 Fig B. Regime Probability and Portfolio Weight Over Time**

목적:

- 모델의 예측 확률이 실제 포트폴리오 비중 변화로 연결되었음을 보여준다.

패널 구성:

- 위: `P(Bear), P(Neutral), P(Bull)`
- 아래: SPY/QQQ/GLD/TLT 비중 변화

이 figure는 Regime-MVO가 단순 표가 아니라 실제 동적 리밸런싱 전략임을 보여준다.

### 4.5 발표용 권장 figure 순서

현재 fig 번호를 그대로 따르기보다, 발표 목적에 맞게 다음 순서로 보여주는 것이 더 자연스럽다.

| 발표 순서 | Figure | 메시지 |
|---:|---|---|
| 1 | Fig 01 Pipeline | HMM pseudo-label → Conv1D+LSTM → Regime-MVO 전체 흐름 |
| 2 | Fig 04 + Fig 05 | 분류기는 완벽하지 않지만 Bear 탐지력이 개선됨 |
| 3 | Fig 03-A Static vs Dynamic Backtest | 정적 benchmark 대비 동적 Regime-MVO의 하방 위험 완화 |
| 4 | Fig 03-B Metric Summary | 수익률 1등이 아니라 MDD/Calmar 관점의 위험관리 전략임 |
| 5 | Fig 06 | Bear 구간에서 위험 완화가 주로 발생함 |
| 6 | Fig 07 | 구성요소별 기여와 EW 1/N benchmark 비교 |
| 7 | Fig 08 | 2022 금리인상형 Bear에서는 실패. 한계와 개선 방향 |

15분 발표에서는 Fig 02 Related Work는 생략하거나 한 장으로 매우 짧게 처리해도 된다.

---

## 5. 15분 시간 배분

| 슬라이드 | 제목 | 시간 |
|---:|---|---:|
| 1 | Title | 0:30 |
| 2 | 연구 배경 | 1:30 |
| 3 | 선행연구 | 1:20 |
| 4 | 연구 질문과 기여 | 0:40 |
| 5 | 전체 파이프라인 | 1:20 |
| 6 | 데이터와 HMM 라벨링 | 1:20 |
| 7 | 모델과 포트폴리오 구성 | 1:20 |
| 8 | 분류 성능 | 1:40 |
| 9 | 최종 백테스트 | 1:40 |
| 10 | 국면별 성과와 Ablation | 1:40 |
| 11 | 2022 하락장 검증 | 1:20 |
| 12 | 한계와 결론 | 1:10 |
| **합계** |  | **15:00** |

---

## 6. 발표자별 분담 예시

| 담당 | 범위 | 시간 | 주요 역할 |
|---|---|---:|---|
| 발표자 1 | 슬라이드 1~3 | 3:20 | 문제의식과 선행연구 |
| 발표자 2 | 슬라이드 4~7 | 4:00 | 연구 질문, 데이터, 방법론 |
| 발표자 3 | 슬라이드 8~10 | 5:00 | 분류 성능과 최종 백테스트 |
| 발표자 4 | 슬라이드 11~12 | 2:40 | 2022 검증, 한계, 결론 |

---

## 7. 참고문헌 후보

발표 슬라이드에는 5개 정도만 넣는 것을 권장한다.

1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*.
   - MVO의 이론적 출발점
   - https://performance-measurement.org/Markowitz1952.pdf

2. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*.
   - regime switching 관점의 대표적 출발점
   - https://www.econometricsociety.org/publications/econometrica/1989/03/01/new-approach-economic-analysis-nonstationary-time-series-and

3. Ang, A., & Bekaert, G. (2002). International Asset Allocation With Regime Shifts. *The Review of Financial Studies*.
   - 국면 변화와 자산배분의 연결
   - https://academic.oup.com/rfs/article/15/4/1137/1568247

4. Dynamic asset allocation for varied financial markets under regime switching framework. (2014). *European Journal of Operational Research*.
   - HMM 기반 국면 식별과 동적 자산배분
   - https://www.sciencedirect.com/science/article/pii/S0377221713002658

5. Costa, G., & Kwon, R. (2020). A Regime-Switching Factor Model for Mean-Variance Optimization. *Journal of Risk*.
   - 국면별 MVO 입력을 사용하는 접근
   - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3609365

6. Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review. *Applied Soft Computing*.
   - 금융 시계열에서 LSTM/딥러닝 사용 배경
   - https://www.sciencedirect.com/science/article/pii/S1568494620301216

---

## 8. 핵심 수식과 개념 노트

이 섹션은 슬라이드에 전부 넣기보다는 발표자 노트 또는 질의응답 대비용으로 사용한다.

### 8.1 HMM label은 pseudo-label

HMM이 만든 국면 label은 관측된 절대적 정답이 아니라, 통계적 모델이 생성한 pseudo-label이다.

$$
\hat{y}^{HMM}_t = g_{\phi}(x_{1:t})
$$

따라서 딥러닝 모델은 실제 시장의 "진짜 정답 국면"을 학습한다기보다 HMM이 정의한 국면 체계를 근사한다.

$$
f_{\theta}(X_t) \approx \hat{y}^{HMM}_{t+h}
$$

발표 표현:

> HMM state는 true label이 아니라 pseudo-label이므로, classification accuracy만으로 경제적 유효성을 주장하지 않고 백테스트로 downstream usefulness를 검증했다.

---

### 8.2 Softmax와 Cross Entropy

모델은 Bear, Neutral, Bull에 대한 logits을 출력한다.

$$
z_i
=
f_{\theta}(X_i)
=
\left[
z_{i,\text{Bear}},
z_{i,\text{Neutral}},
z_{i,\text{Bull}}
\right]
$$

이를 softmax로 확률화한다.

$$
p_{i,k}
=
\frac{\exp(z_{i,k})}
{\sum_{j=1}^{3}\exp(z_{i,j})}
$$

분류 손실은 weighted cross entropy이다.

$$
\mathcal{L}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^{N}
w_{y_i}
\log p_{i,y_i}
$$

클래스 가중치는 대략 클래스 빈도에 반비례한다.

$$
w_k
=
\frac{N}{K n_k}
$$

Neutral은 추가로 `neutral_boost=1.2`를 적용했다.

발표 표현:

> 우리 모델은 수익률 값을 회귀한 것이 아니라 HMM pseudo-label을 분류했기 때문에 MSE가 아니라 weighted cross entropy를 사용했다.

---

### 8.3 MVO의 의미

MVO는 기대수익률과 위험의 trade-off를 최적화한다.

$$
\max_w
\left(
w^\top \mu
-
\lambda w^\top \Sigma w
\right)
$$

여기서

- `w`: 포트폴리오 비중
- `μ`: 기대수익률 벡터
- `Σ`: 공분산 행렬
- `λ`: 위험 회피 계수

중요한 점은 `μ`와 `Σ`를 실제로 아는 것이 아니라 과거 훈련 데이터에서 추정한다는 것이다.

$$
\hat{\mu}
=
\frac{1}{T}
\sum_{t=1}^{T} r_t
$$

$$
\hat{\Sigma}
=
\operatorname{Cov}(r_1, r_2, \dots, r_T)
$$

발표 표현:

> MVO의 기대수익률은 미래를 아는 값이 아니라 과거 데이터로부터 추정한 값이기 때문에, 시장 국면이 바뀌면 정적 MVO가 불안정해질 수 있다.

---

### 8.4 Regime-MVO의 동적 비중

국면별 MVO 비중을 먼저 계산한다.

$$
w_{\text{Bear}},
\quad
w_{\text{Neutral}},
\quad
w_{\text{Bull}}
$$

모델이 예측한 국면 확률은 다음과 같다.

$$
p_t
=
\left[
p_{t,\text{Bear}},
p_{t,\text{Neutral}},
p_{t,\text{Bull}}
\right]
$$

최종 포트폴리오 비중은 확률 가중 평균이다.

$$
w_t
=
p_{t,\text{Bear}} w_{\text{Bear}}
+
p_{t,\text{Neutral}} w_{\text{Neutral}}
+
p_{t,\text{Bull}} w_{\text{Bull}}
$$

발표 표현:

> Regime-MVO는 오늘의 국면을 하나로 단정하는 hard switching이 아니라, 모델의 불확실성을 반영해 국면별 포트폴리오를 확률적으로 섞는 soft allocation 방식이다.

---

### 8.5 정적 포트폴리오 vs 동적 포트폴리오

정적 포트폴리오는 테스트 기간 동안 비중이 바뀌지 않는다.

$$
w_t = w_{\text{static}}
$$

예:

- Buy & Hold: SPY 100%
- EW 1/N: SPY/QQQ/GLD/TLT 각각 25%
- 60/40: 사전에 정한 60%, 40% 비중 유지
- Regime-Agnostic MVO: 전체 훈련 구간으로 한 번 계산한 MVO 비중 유지

동적 포트폴리오는 매 시점의 예측 확률에 따라 비중이 바뀐다.

$$
w_t = f(p_t)
$$

발표 표현:

> 본 프로젝트의 비교는 정적 benchmark와 동적 Regime-MVO의 비교이며, 핵심은 동적 비중 조절이 drawdown을 줄였는지이다.

---

### 8.6 백테스트 성과 지표

포트폴리오 수익률:

$$
r_{p,t+1}
=
w_t^\top r_{t+1}
$$

누적수익률:

$$
R_T
=
\prod_{t=1}^{T}
(1+r_{p,t}) - 1
$$

Sharpe ratio:

$$
\text{Sharpe}
\approx
\frac{\mathbb{E}[r_p]}
{\sigma(r_p)}
$$

MDD:

$$
\text{MDD}
=
\max_t
\left(
\frac{
\max_{\tau \leq t} V_{\tau} - V_t
}{
\max_{\tau \leq t} V_{\tau}
}
\right)
$$

Calmar:

$$
\text{Calmar}
=
\frac{\text{Annualized Return}}
{|\text{MDD}|}
$$

발표 표현:

> Regime-MVO는 누적수익률 1등은 아니지만, MDD를 -7.2%로 낮추고 Calmar를 2.16으로 개선했다. 따라서 본 전략의 의의는 수익률 극대화보다 위험 조정 성과에 있다.

---

### 8.7 Transformer 관련 설명

Transformer self-attention은 모든 시점 간 관계를 학습한다.

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

장점:

- 모든 시점 간 pairwise relation 학습 가능
- 충분한 데이터와 긴 시계열에서는 강력함

본 프로젝트에서 최종 채택하지 않은 이유:

- 금융 데이터는 noise가 크다.
- 최종 train sample은 488개로 작다.
- 입력 길이도 30일로 길지 않다.
- 표현력이 큰 모델은 pseudo-label과 noise에 과적합할 위험이 있다.
- Conv1D+LSTM은 국소 패턴 추출과 순차적 정보 누적이라는 inductive bias가 있어 더 보수적이다.

발표 표현:

> Transformer도 검토했지만, 본 데이터 조건에서는 self-attention의 높은 표현력보다 Conv1D+LSTM의 시계열 inductive bias가 더 안정적이라고 판단했다.

---

## 9. 예상 Q&A

**Q1. 왜 수익률이 가장 높은 EW 1/N이 아니라 Regime-MVO를 최종 전략으로 보나?**

EW 1/N은 테스트 구간에서 누적수익률과 Sharpe가 좋지만, 본 프로젝트의 목표는 수익률 1등 전략을 찾는 것이 아니라 국면 정보가 포트폴리오 위험 조절에 유용한지 검증하는 것이다. Regime-MVO는 누적수익률에서는 EW 1/N보다 낮지만, MDD -7.2%, Calmar 2.16으로 하방 위험 관리 측면에서 의미가 있다. 특히 국면을 무시한 Regime-Agnostic MVO의 MDD가 -20.8%였다는 점과 비교하면, 국면 conditioning이 위험 노출을 줄였다는 해석이 가능하다.

**Q2. Neutral Recall이 0%면 모델이 실패한 것 아닌가?**

Neutral 예측은 분명한 한계로 인정해야 한다. 다만 Neutral은 HMM Sharpe ranking상 Bear와 Bull 사이의 중간 상태라 경계가 모호하고, pseudo-label 자체의 불확실성이 크다. 본 프로젝트의 포트폴리오 목적에서는 Neutral을 정확히 맞히는 것보다 Bear 위험 구간을 놓치지 않는 것이 더 중요했고, Bear Recall은 34.9%에서 60.5%로 개선됐다. 따라서 모델은 완전한 국면 분류기로는 부족하지만, 하방 위험 관리 신호로는 일정한 유용성을 보였다.

**Q3. 2022년에는 Regime-MVO가 왜 안 좋았나?**

훈련 구간에서 Bear MVO가 TLT 100%를 선택했는데, 2022년은 금리인상으로 주식과 장기채가 동시에 하락한 특수한 Bear였다. 이 때문에 TLT 중심 방어가 실패했고, 현금을 사용할 수 있는 DL Regime SPY/Cash가 더 잘 버텼다.

**Q4. HMM 라벨이 정답이라고 볼 수 있나?**

완전한 정답은 아니다. HMM 라벨은 숨겨진 시장 국면을 통계적으로 추정한 pseudo-label이다. 따라서 이 프로젝트는 "진짜 국면 정답을 맞히는 문제"라기보다, 일관된 국면 정의를 만들고 그 국면 신호가 포트폴리오에 유용한지 검증한 작업이다. 그래서 classification accuracy만 제시하지 않고, 예측된 국면 확률을 실제 리밸런싱에 연결한 백테스트 결과를 함께 제시했다.

**Q5. 왜 가격이나 수익률을 직접 예측하지 않았나?**

금융 가격을 직접 예측하는 것은 노이즈가 크고 과적합 위험이 높다. 대신 Bear/Neutral/Bull 같은 국면을 예측하면 해석 가능성이 높고, 포트폴리오 리스크 조절과 직접 연결하기 쉽다.

**Q6. 왜 end-to-end로 수익률이나 Sharpe를 직접 최적화하지 않았나?**

가능한 접근이지만, 본 프로젝트에서는 데이터 수가 제한적이고 금융 수익률의 noise가 크기 때문에 train 구간 수익률에 과최적화될 위험이 크다고 판단했다. Sharpe나 MDD 같은 지표는 cross entropy보다 gradient가 불안정할 수 있고, 거래비용과 리밸런싱 제약까지 포함하면 최적화가 더 어려워진다. 따라서 본 프로젝트는 국면 예측 모듈과 포트폴리오 배분 모듈을 분리하여 해석 가능성을 유지하고, 최종 경제적 유효성은 백테스트로 검증했다.

**Q7. Transformer를 쓰면 더 좋지 않았나?**

Transformer 계열 구조도 검토되었다. 0528 수정 결과의 모델 구조 문서에는 SPY 기반 Transformer 구조가 정리되어 있고, 노트북에도 Transformer 실험 흔적이 있다. 다만 본 프로젝트의 최종 데이터는 sample 수가 작고 금융 noise가 크며 입력 길이도 30일로 짧다. Transformer는 모든 시점 간 pairwise relation을 학습할 수 있어 표현력이 크지만, 이 조건에서는 noise를 과적합할 위험도 커진다. 반면 Conv1D+LSTM은 국소 패턴 추출과 순차적 정보 누적이라는 inductive bias가 있어 더 보수적이고 안정적인 선택이었다.

**Q8. HMM state를 예측하는 것이 실제 market prediction이라고 할 수 있나?**

엄밀히 말하면 HMM state prediction 자체를 market prediction이라고 강하게 주장하기는 어렵다. HMM state는 absolute market truth가 아니라 pseudo-label이기 때문이다. 그래서 본 프로젝트의 주장은 "HMM state를 잘 맞혔다"에서 끝나지 않는다. 핵심은 "그 state probability를 포트폴리오 리밸런싱에 사용했을 때 정적 benchmark 대비 하방 위험이 줄었는가"이며, 이 부분을 백테스트로 검증했다.

**Q9. Regime-Agnostic MVO가 수익률은 가장 높은데 왜 문제인가?**

Regime-Agnostic MVO는 훈련 데이터 전체로 한 번 계산한 MVO 비중을 테스트 기간에 고정한 전략이다. 누적수익률은 64.8%로 높았지만 MDD가 -20.8%로 크게 악화됐다. 즉 높은 수익률을 얻는 대신 큰 drawdown을 감수한 것이다. 본 프로젝트는 하락 위험 완화에 초점을 두기 때문에, Regime-MVO의 MDD -7.2%가 더 중요한 결과로 해석된다.
